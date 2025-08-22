{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // Node label distribution\
MATCH (n)\
UNWIND labels(n) AS label\
RETURN label, count(*) AS count\
ORDER BY count DESC;\
\
// Node degree distribution\
MATCH (n)\
OPTIONAL MATCH (n)-[r]-()\
WITH n, count(r) AS degree\
RETURN degree, count(*) AS nodes_with_degree\
ORDER BY degree;\
\
// Top connected nodes\
MATCH (n)\
OPTIONAL MATCH (n)-[r]-()\
WITH n, count(r) AS degree\
RETURN labels(n) AS node_labels, n AS node, degree\
ORDER BY degree DESC\
LIMIT 20;\
\
// Relationship type distribution per source\
MATCH ()-[r]->()\
WHERE r.source IS NOT NULL\
RETURN type(r) AS relationship_type, r.source AS source, count(*) AS count\
ORDER BY relationship_type, count DESC;\
\
// Triples per source \
MATCH (subject)-[predicate]->(object)\
WHERE predicate.source IS NOT NULL\
RETURN predicate.source AS source,\
       labels(subject) AS subject_labels,\
       type(predicate) AS predicate_type,\
       labels(object) AS object_labels,\
       count(*) AS triple_count\
ORDER BY source, triple_count DESC;\
\
// Unique entities per source (with fuzzy matching)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 MATCH (n)-[r]->()\
WHERE r.source IS NOT NULL AND n.name IS NOT NULL\
WITH r.source AS source, labels(n) AS node_type, n.name AS raw\
\
// 1) Normalize\
WITH source, node_type, raw, apoc.text.clean(toLower(raw)) AS norm\
\
// 2) Build fuzzy key: phonetic + light de-noising + short stem\
WITH source, node_type, raw, norm,\
     apoc.text.phonetic(norm) AS metaphone,\
     apoc.text.regreplace(norm, '\\\\b(disease|syndrome|disorder)\\\\b', '') AS denoised\
WITH source, node_type, raw, metaphone,\
     apoc.text.regreplace(denoised, '\\\\s+', ' ') AS denoised2\
WITH source, node_type, raw,\
     coalesce(metaphone,'') + ':' + left(denoised2, 10) AS fuzzy_key\
\
// 3) Collapse names per fuzzy bucket\
WITH source, node_type, fuzzy_key, collect(DISTINCT raw) AS name_variants\
\
// ---- NEW: 3b) Keep only variants close to the seed by edit distance / Jaro-Winkler ----\
WITH source, node_type, fuzzy_key,\
     apoc.coll.sort(name_variants)[0] AS seed,\
     name_variants\
// thresholds: tweak as needed\
WITH source, node_type, fuzzy_key, seed,\
     [v IN name_variants\
      WHERE apoc.text.distance(toLower(v), toLower(seed)) <= 2\
         OR apoc.text.jaroWinklerDistance(toLower(v), toLower(seed)) >= 0.92] AS tight_variants\
\
// 4) Choose a stable representative (lexicographically first of the tight set)\
WITH source, node_type, fuzzy_key, apoc.coll.sort(tight_variants)[0] AS canonical_name\
\
// 5) Keep buckets that appear in exactly one source\
WITH node_type, fuzzy_key, canonical_name, collect(DISTINCT source) AS sources\
WHERE size(sources) = 1\
\
RETURN\
  sources[0]     AS unique_to_source,\
  node_type      AS entity_type,\
  canonical_name AS unique_entity,\
  fuzzy_key      AS fuzzy_bucket\
ORDER BY unique_to_source, entity_type, unique_entity;\
\
\
\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \
}