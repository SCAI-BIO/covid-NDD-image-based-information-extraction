{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // 6. Therapeutic and Drug Target Analysis\
\
// Find entities that could be therapeutic targets (highly connected to both domains)\
MATCH (target)-[r1]->(covid_related)\
MATCH (target)-[r2]->(neuro_related)\
WHERE any(kw IN ['covid', 'sars-cov-2', 'cytokine storm', 'inflammation'] \
    WHERE toLower(covid_related.name) CONTAINS toLower(kw))\
AND any(kw IN ['neurodegeneration', 'tau', 'amyloid', 'neuroinflammation', 'cognitive'] \
    WHERE toLower(neuro_related.name) CONTAINS toLower(kw))\
AND (target:Gene OR target:Protein OR target:Chemical)\
RETURN \
    labels(target) AS target_type,\
    target.name AS potential_target,\
    collect(DISTINCT covid_related.name) AS covid_connections,\
    collect(DISTINCT neuro_related.name) AS neuro_connections,\
    collect(DISTINCT r1.source) AS evidence_sources\
ORDER BY size(covid_connections) + size(neuro_connections) DESC\
LIMIT 10;\
\
\
// Find drugs that affect COVID-related processes and might help neurodegeneration\
MATCH (drug)-[r1]->(covid_process)-[r2]->(neuro_outcome)\
WHERE (drug:Chemical OR drug.name CONTAINS 'drug' OR drug.name CONTAINS 'therapy')\
AND any(kw IN ['covid', 'viral', 'cytokine', 'inflammation', 'immune'] \
    WHERE toLower(covid_process.name) CONTAINS toLower(kw))\
AND any(kw IN ['neurodegeneration', 'cognitive', 'memory', 'neuroprotection'] \
    WHERE toLower(neuro_outcome.name) CONTAINS toLower(kw))\
RETURN \
    drug.name AS potential_drug,\
    covid_process.name AS covid_target,\
    type(r1) AS drug_effect,\
    r1.source AS evidence_source1,\
    neuro_outcome.name AS neuro_benefit,\
    type(r2) AS mechanism,\
    r2.source AS evidence_source2\
ORDER BY drug.name\
\
\
\
// 6.1 Potential Therapeutic Targets\
// Entities connected to both COVID- and neuro-related nodes via regulatory relationships\
MATCH (target)-[r1]->(covid_related)\
MATCH (target)-[r2]->(neuro_related)\
WHERE any(kw IN ['covid','sars-cov-2','cytokine storm','inflammation'] \
          WHERE toLower(coalesce(covid_related.name,'')) CONTAINS toLower(kw))\
  AND any(kw IN ['neurodegeneration','tau','amyloid','neuroinflammation','cognitive'] \
          WHERE toLower(coalesce(neuro_related.name,'')) CONTAINS toLower(kw))\
  AND (target:Gene OR target:Protein OR target:Chemical)\
  AND type(r1) IN ['INHIBITS','ACTIVATES','REGULATES','MODULATES']\
  AND type(r2) IN ['INHIBITS','ACTIVATES','REGULATES','MODULATES']\
RETURN \
    labels(target) AS target_type,\
    target.name AS potential_target,\
    collect(DISTINCT covid_related.name) AS covid_connections,\
    collect(DISTINCT neuro_related.name) AS neuro_connections,\
    collect(DISTINCT r1.source) AS evidence_sources\
ORDER BY size(covid_connections) + size(neuro_connections) DESC\
LIMIT 10;\
\
// 6.2 Drug Repurposing Opportunities\
// Drugs acting on COVID-related processes with neuro-relevant outcomes\
MATCH (drug)-[r1]->(covid_process)-[r2]->(neuro_outcome)\
WHERE (drug:Chemical OR toLower(coalesce(drug.name,'')) CONTAINS 'drug' OR toLower(coalesce(drug.name,'')) CONTAINS 'therapy')\
  AND any(kw IN ['covid','viral','cytokine','inflammation','immune'] \
          WHERE toLower(coalesce(covid_process.name,'')) CONTAINS toLower(kw))\
  AND any(kw IN ['neurodegeneration','cognitive','memory','neuroprotection'] \
          WHERE toLower(coalesce(neuro_outcome.name,'')) CONTAINS toLower(kw))\
RETURN \
    drug.name AS potential_drug,\
    covid_process.name AS covid_target,\
    type(r1) AS drug_effect,\
    r1.source AS evidence_source1,\
    neuro_outcome.name AS neuro_benefit,\
    type(r2) AS mechanism,\
    r2.source AS evidence_source2\
ORDER BY potential_drug;\
}