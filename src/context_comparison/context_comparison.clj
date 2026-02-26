(ns user
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [conexp.fca.contexts :as contexts]
            [conexp.fca.lattices :as lattices]
            [conexp.io.contexts :as io-contexts]))

;; Utilities
(defn ensure-dir! [^String dir]
  (let [f (io/file dir)]
    (.mkdirs f)
    (.getAbsolutePath f)))

(defn basename-noext [^String path]
  (let [f (io/file path)
        name (.getName f)
        dot (.lastIndexOf name ".")]
    (if (neg? dot) name (subs name 0 dot))))

;; Read a Burmeister (.cxt) context
(defn read-burmeister-context
  "Reads a Burmeister FCA context (.cxt) from disk using conexp-clj."
  [^String path]
  (io-contexts/read-context path))

(defn all-concepts
  "Enumerate concepts using TITANIC with threshold 0.0 (i.e., full lattice). Returns vector of [extent intent]."
  [ctx]
  (let [intents (lattices/titanic-iceberg-intent-seq ctx 0.0)]
    (mapv (fn [intent]
            [(contexts/attribute-derivation ctx intent) intent])
          intents)))

(defn concept-supports
  "Return a vector of supports (|extent| / |G|) for concepts."
  [concepts n-objects]
  (mapv (fn [[extent _]] (/ (double (count extent)) (double n-objects))) concepts))

(defn quantiles [xs]
  (let [s (sort xs)
        n (count s)
        pick (fn [p]
               (if (zero? n)
                 0.0
                 (let [idx (min (dec n) (int (Math/floor (* p (dec n)))))]
                   (nth s idx))))]
    {:q0 (first s)
     :q1 (pick 0.25)
     :q2 (pick 0.50)
     :q3 (pick 0.75)
     :q4 (last s)}))

(defn save-edn [x ^String path]
  (spit path (pr-str x))
  path)

(defn save-json [x ^String path]
  (with-open [w (io/writer path)]
    (json/write x w))
  path)

(defn analyze-context
  "Analyze a Burmeister context at `cxt-path`. Writes:
   - EDN concepts to results/context_comparison/<basename>_concepts.edn
   - JSON stats to results/context_comparison/<basename>_stats.json

  Returns the stats map. Optionally accepts `min-support` to restrict to iceberg concepts; defaults to 0.0 (full lattice)."
  ([^String cxt-path]
   (analyze-context cxt-path 0.0))
  ([^String cxt-path min-support]
   (let [out-dir (ensure-dir! "results/context_comparison")
         base (basename-noext cxt-path)
         ctx (read-burmeister-context cxt-path)
         nG (count (contexts/objects ctx))
         nM (count (contexts/attributes ctx))
         intents (lattices/titanic-iceberg-intent-seq ctx (double (or min-support 0.0)))
         concepts (mapv (fn [intent]
                          [(contexts/attribute-derivation ctx intent) intent])
                        intents)
         supports (concept-supports concepts nG)
         supp-stats {:min (when (seq supports) (apply min supports))
                     :max (when (seq supports) (apply max supports))
                     :mean (when (seq supports) (/ (reduce + supports) (double (count supports))))
                     :median (:q2 (quantiles supports))
                     :quantiles (quantiles supports)}
         stats {:context-path cxt-path
                :objects nG
                :attributes nM
                :num-concepts (count concepts)
                :min-support (double (or min-support 0.0))
                :support-stats supp-stats}
         edn-path (str out-dir "/" base "_concepts.edn")
         json-path (str out-dir "/" base "_stats.json")]
     (save-edn concepts edn-path)
     (save-json stats json-path)
     (println "Analyzed context:" cxt-path
              "| G=" nG "M=" nM
              "| concepts=" (count concepts)
              "| min-support=" (double (or min-support 0.0))
              "| saved EDN to" edn-path
              "| saved stats to" json-path)
     stats)))
