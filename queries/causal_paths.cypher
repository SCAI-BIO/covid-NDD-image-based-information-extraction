{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 //Potential Causal Chains Between Entities\
MATCH path = (start)-[:CAUSES|LEADS_TO*2..4]->(end)\
WHERE any(kw IN ['covid', 'spike', 'cytokine'] WHERE toLower(start.name) CONTAINS kw)\
  AND any(kw IN ['neuro', 'tau', 'amyloid', 'cognitive'] WHERE toLower(end.name) CONTAINS kw)\
RETURN path\
LIMIT 10\
\
\
// 7. Temporal and Causal Chain Analysis\
\
// 7.1 Multi-step Causal Chains (3 hops)\
// Causal cascade from COVID-related to neurodegeneration-related entities\
MATCH path = (covid)-[r1]->(intermediate1)-[r2]->(intermediate2)-[r3]->(neuro)\
WHERE any(kw IN ['covid','sars-cov-2','viral entry'] \
          WHERE toLower(coalesce(covid.name,'')) CONTAINS toLower(kw))\
  AND any(kw IN ['neurodegeneration','cognitive decline','memory loss','dementia'] \
          WHERE toLower(coalesce(neuro.name,'')) CONTAINS toLower(kw))\
  AND all(r IN [r1,r2,r3] WHERE type(r) IN ['CAUSES','LEADS_TO','INDUCES','TRIGGERS','RESULTS_IN'])\
RETURN \
    covid.name AS initial_trigger,\
    type(r1) AS step1, r1.source AS source1, intermediate1.name AS mechanism1,\
    type(r2) AS step2, r2.source AS source2, intermediate2.name AS mechanism2,\
    type(r3) AS step3, r3.source AS source3, neuro.name AS final_outcome\
LIMIT 10;\
\
// 7.2 GPT's Unique Causal Insights\
// Causal relationships present only in GPT, absent in CBM\
MATCH (a)-[r_gpt \{source: 'GPT'\}]->(b)\
WHERE type(r_gpt) IN ['CAUSES','LEADS_TO','INDUCES','TRIGGERS','RESULTS_IN']\
  AND NOT EXISTS \{\
    MATCH (a)-[r_manual \{source: 'CBM'\}]->(b)\
    WHERE type(r_manual) IN ['CAUSES','LEADS_TO','INDUCES','TRIGGERS','RESULTS_IN','INCREASES','DECREASES']\
  \}\
RETURN \
    labels(a) AS cause_type,\
    a.name AS cause,\
    type(r_gpt) AS causal_relationship,\
    labels(b) AS effect_type,\
    b.name AS effect,\
    count(*) AS frequency\
ORDER BY frequency DESC\
LIMIT 20;\
\
// Find causal mechanisms that GPT identified but manual curation missed\
MATCH (a)-[r_gpt \{source: 'GPT'\}]->(b)\
WHERE NOT EXISTS \{\
    MATCH (a)-[r_manual \{source: 'CBM'\}]->(b)\
    WHERE type(r_manual) = type(r_gpt)\
\}\
AND (a:BiologicalProcess OR b:BiologicalProcess OR \
     type(r_gpt) IN ['CAUSES', 'LEADS_TO', 'INDUCES', 'TRIGGERS', 'ACTIVATES'])\
RETURN \
    labels(a) AS subject_type,\
    a.name AS subject,\
    type(r_gpt) AS relationship,\
    labels(b) AS object_type,\
    b.name AS object,\
    count(*) AS frequency\
ORDER BY frequency DESC\
LIMIT 15\
\
// Find 3-step causal chains from COVID to neurodegeneration\
MATCH path = (covid)-[r1]->(intermediate1)-[r2]->(intermediate2)-[r3]->(neuro)\
WHERE any(kw IN ['covid', 'sars-cov-2', 'viral entry'] \
    WHERE toLower(covid.name) CONTAINS toLower(kw))\
AND any(kw IN ['neurodegeneration', 'cognitive decline', 'memory loss', 'dementia'] \
    WHERE toLower(neuro.name) CONTAINS toLower(kw))\
AND all(r IN [r1, r2, r3] WHERE type(r) IN ['CAUSES', 'LEADS_TO', 'INDUCES', 'TRIGGERS', 'RESULTS_IN'])\
RETURN \
    covid.name AS initial_trigger,\
    type(r1) AS step1,\
    r1.source AS source1,\
    intermediate1.name AS mechanism1,\
    type(r2) AS step2,\
    r2.source AS source2,\
    intermediate2.name AS mechanism2,\
    type(r3) AS step3,\
    r3.source AS source3,\
    neuro.name AS final_outcome\
LIMIT 10\
\
}