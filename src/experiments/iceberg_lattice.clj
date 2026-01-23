(ns user
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
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
  (let [objects (:index ctx-json)
        attributes (:columns ctx-json)
        raw-incidence (:data ctx-json)
        incidence (mapv (fn [row]
                          (mapv (fn [x]
                                  (normalize-bit x))
                                row))
                        raw-incidence)]
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
         saved-path (save-iceberg-concepts concepts output-path)]
     (println "Loaded context from" context-path
              "| objects:" (count (:index ctx-json))
              "| attributes:" (count (:columns ctx-json))
              "| iceberg concepts:" (count concepts)
              "| saved to:" saved-path)
    ;;  concepts
     )))
