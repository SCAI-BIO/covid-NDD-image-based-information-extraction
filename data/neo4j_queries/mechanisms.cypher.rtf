{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // 5. Mechanistic Category Analysis\
\
// 5.1 GPT's Mechanistic Category Distribution\
// Analyze mechanistic categories identified by each source.\
// Optional parameter: $min_count (default 5)\
WITH coalesce($min_count, 5) AS min_count\
MATCH ()-[r]->()\
WHERE r.source IS NOT NULL AND r.mechanism IS NOT NULL\
WITH r.source AS source, r.mechanism AS mechanism, count(*) AS frequency\
WITH source, collect(\{mechanism: mechanism, count: frequency\}) AS mechanisms\
RETURN \
    source,\
    [m IN mechanisms WHERE m.count >= min_count | m.mechanism] AS frequent_mechanisms,\
    size([m IN mechanisms WHERE m.count >= min_count]) AS frequent_mechanism_count,\
    size(mechanisms) AS total_unique_mechanisms\
ORDER BY source;\
\
// 5.2 Cross-Source Mechanism Validation\
// Find mechanisms identified by both GPT and manual curation (CBM)\
MATCH ()-[r1 \{source: 'GPT'\}]->()\
MATCH ()-[r2 \{source: 'CBM'\}]->()\
WHERE r1.mechanism IS NOT NULL AND r2.mechanism IS NOT NULL\
WITH r1.mechanism AS mechanism, count(r1) AS gpt_count, count(r2) AS manual_count\
WHERE gpt_count > 0 AND manual_count > 0\
RETURN \
    mechanism,\
    gpt_count,\
    manual_count,\
    abs(gpt_count - manual_count) AS count_difference,\
    (gpt_count + manual_count) AS total_evidence\
ORDER BY total_evidence DESC, count_difference ASC;\
}