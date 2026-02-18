(ns user
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [conexp.fca.contexts :as contexts]
            [conexp.fca.lattices :as lattices]))

;; This file is designed to be loaded from a REPL or from Python via `(load-file ...)`.
;; After loading, call `run-iceberg` to compute iceberg concepts from the JSON context.

(def default-context-path
  "Default path to the FCA context exported from Python (pandas `to_json` with orient=split)."
  "resources/banksearch/fca_topic_model_context.json")

(def default-output-path
  "Default path where iceberg concepts are saved as EDN."
  "resources/banksearch/iceberg_concepts.edn")

(defn read-context-json
  "Reads the FCA context JSON from disk.

  The expected JSON format is pandas `orient=split` and contains:
  - `index`: object identifiers (rows)
  - `columns`: attribute identifiers (columns)
  - `data`: 2D array of booleans/0-1 values

  Returns a map with keyword keys: `:index`, `:columns`, `:data`."
  [^String path]
  (with-open [r (io/reader path)]
    (json/read r :key-fn keyword)))

(defn normalize-bit
  "Normalizes a JSON entry into 0 or 1 when possible.
  Returns nil when the value is not a valid bit."
  [x]
  (cond
    (true? x) 1
    (false? x) 0
    (= x 1) 1
    (= x 0) 0
    (= x 1.0) 1
    (= x 0.0) 0
    (= x "1") 1
    (= x "0") 0
    (= x "true") 1
    (= x "false") 0
    :else nil))

(defn first-invalid-entry
  "Finds the first invalid entry in the incidence matrix.
  Returns a map with row/col/value, or nil if all entries are valid."
  [raw-incidence normalized-incidence]
  (first
   (for [row-idx (range (count normalized-incidence))
         col-idx (range (count (nth normalized-incidence row-idx)))
         :let [normalized (get-in normalized-incidence [row-idx col-idx])
               raw (get-in raw-incidence [row-idx col-idx])]
         :when (nil? normalized)]
     {:row row-idx :col col-idx :value raw :type (type raw)})))

(defn report-invalid-entry
  "Prints a detailed error message for the first invalid entry."
  [bad]
  (binding [*out* *err*]
    (println "Invalid incidence entry (expected 0/1 or true/false)."
             "row=" (:row bad)
             "col=" (:col bad)
             "value=" (:value bad)
             "type=" (:type bad))))

(defn context-from-json
  "Converts a parsed JSON context map into a ConExp FCA context.

  Parameters:
  - ctx-json: Map with keys `:index`, `:columns`, `:data` as returned by `read-context-json`.

  Returns:
  A ConExp context suitable for lattice computations."
  [ctx-json]
  ;; objects G = topics = rows 
  ;; attributes M = documents = columns
  (let [objects (:index ctx-json)
        attributes (:columns ctx-json)
        raw-incidence (:data ctx-json)
        incidence (mapv (fn [row]
                          (mapv (fn [x]
                                  (normalize-bit x))
                                row))
                        raw-incidence)]
        ;; Debugging info and sanity checks
        ;; (println "Ones per row:::" (mapv #(count (filter identity %)) incidence))
        (let [flat (mapcat identity incidence)]
          (println "Total entries:" (count flat))
          (println "Ones:" (count (filter #(== 1 %) flat)))
          (println "Zeros:" (count (filter zero? flat))))
        (println "Distinct normalized values:" (set (mapcat identity incidence)))

    (when-let [bad (first-invalid-entry raw-incidence incidence)]
      (report-invalid-entry bad)
      (throw (ex-info "Invalid incidence entry (expected 0/1 or true/false)." bad)))
    (try
      (do
      (let [flat-incidence (vec (mapcat identity incidence))]
        (println "Creating context from"
                 (count objects) "objects and"
                 (count attributes) "attributes..."
                 "| flattened incidence length:" (count flat-incidence))
      (contexts/make-context-from-matrix objects attributes flat-incidence)))
      (catch AssertionError e
        (when-let [bad (first-invalid-entry raw-incidence incidence)]
          (report-invalid-entry bad))
        (throw e)))))

(defn iceberg-concepts
  "Computes iceberg concepts for a context and minimum support threshold.

  Parameters:
  - ctx: FCA context.
  - min-support: Minimum support threshold in [0, 1].

  Returns:
  A vector of concepts, each in the form [extent intent]."
  [ctx min-support]
  (println "Computing iceberg concepts with min-support =" min-support "...")
  (let [intents (lattices/titanic-iceberg-intent-seq ctx (double min-support))]
    (mapv (fn [intent]
            [(contexts/attribute-derivation ctx intent) intent])
          intents)))

(defn save-iceberg-concepts
  "Saves iceberg concepts to disk as EDN.

  Parameters:
  - concepts: Vector of [extent intent] pairs from `iceberg-concepts`.
  - output-path: File path to write the EDN data.

  Returns:
  The output path as confirmation."
  [concepts ^String output-path]
  (spit output-path (pr-str concepts))
  output-path)

(defn get-iceberg-context
  "Rows = iceberg concepts, cols = attributes, cell=1 iff attribute in concept intent."
  [iceberg-concepts]
  (let [columns (->> iceberg-concepts (map second) (mapcat identity) distinct sort vec)
;;         index   (mapv #(str "c" %) (range (count iceberg-concepts)))
        index   (mapv first iceberg-concepts)   ;; preserve original ids/extents
        data    (mapv (fn [[_ intent]]
                        (let [intent-set (set intent)]
                          (mapv (fn [a] (if (contains? intent-set a) 1 0)) columns)))
                      iceberg-concepts)]
    {:index index :columns columns :data data}))

(defn save-context-json
  [ctx-json ^String output-path]
  (with-open [w (io/writer output-path)]
    (json/write ctx-json w))
  output-path)

(defn csv-escape
  [v]
  (let [s (cond
            (string? v) v
            :else (pr-str v))]
    (str "\"" (str/replace s "\"" "\"\"") "\"")))

(defn save-context-csv
  [ctx-json ^String output-path]
  (let [header (str/join "," (cons (csv-escape "index")
                                   (map csv-escape (:columns ctx-json))))
        rows (map (fn [idx row]
                    (str/join "," (cons (csv-escape idx)
                                        (map csv-escape row))))
                  (:index ctx-json)
                  (:data ctx-json))
        content (str (str/join "\n" (cons header rows)) "\n")]
    (spit output-path content))
  output-path)


(defn run-iceberg
  "Public entry point for Python usage.

  Usage:
  - (run-iceberg 0.9) ; uses `default-context-path` and `default-output-path`
  - (run-iceberg \"path/to/context.json\" 0.9)
  - (run-iceberg \"path/to/context.json\" 0.9 \"path/to/output.edn\")

  Returns:
  A vector of iceberg concepts. Prints a short summary and writes the concepts to disk."
  ([min-support]
   (run-iceberg default-context-path min-support default-output-path))
  ([^String context-path min-support]
   (run-iceberg context-path min-support default-output-path))
  ([^String context-path min-support ^String output-path]
   (let [ctx-json (read-context-json context-path)
         ctx (context-from-json ctx-json)
         concepts (iceberg-concepts ctx min-support)
         iceberg-context (get-iceberg-context concepts)
         saved-path (save-iceberg-concepts concepts output-path)
         cxt-save-path (save-context-json iceberg-context
                                          "resources/banksearch/topic_model/iceberg_context.json")
         cxt-save-path-csv (save-context-csv iceberg-context
                                             "resources/banksearch/topic_model/iceberg_context.csv")
        ]
     (println "Loaded context from" context-path
              "| min support:" min-support
              "| objects:" (count (:index ctx-json))
              "| attributes:" (count (:columns ctx-json))
              "| iceberg concepts:" (count concepts)
              "| concepts saved to:" saved-path
              "| context saved to:" cxt-save-path
              "| context csv saved to:" cxt-save-path-csv)
    ;;  concepts
     )))
