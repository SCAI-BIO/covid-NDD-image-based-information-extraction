{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // COVID vs Neurodegeneration emphasis\
MATCH (s)-[r \{source:"GPT"\}]->(o)\
WHERE toLower(s.name) CONTAINS "covid" OR toLower(o.name) CONTAINS "covid"\
RETURN "COVID" AS Focus, count(*) AS Mentions\
UNION ALL\
MATCH (s)-[r \{source:"GPT"\}]->(o)\
WHERE toLower(s.name) CONTAINS "alzheimer"\
   OR toLower(s.name) CONTAINS "parkinson"\
   OR toLower(s.name) CONTAINS "neurodegeneration"\
   OR toLower(o.name) CONTAINS "alzheimer"\
   OR toLower(o.name) CONTAINS "parkinson"\
   OR toLower(o.name) CONTAINS "neurodegeneration"\
RETURN "Neurodegeneration" AS Focus, count(*) AS Mentions;\
\
//All Simple Paths Between COVID and Neuro Nodes\
MATCH (start), (end)\
WHERE any(kw IN [\
    'covid', 'coronavirus', 'sars-cov-2', 'long covid', 'pandemic',\
    'spike protein', 'ace2', 'tmprss2', 'ifitm3', 'il6', 'il1b', 'tnf', 'cxcl10', 'tlr7',\
    'epithelial cell', 't cell', 'b cell', 'monocyte', 'alveolar macrophage',\
    'remdesivir', 'dexamethasone', 'cytokine', 'interferon',\
    'cytokine storm', 'viral entry', 'immune response', 'hypoxia',\
    'fever', 'cough', 'ards', 'dyspnea', 'fatigue', 'thrombosis'\
] WHERE toLower(start.name) CONTAINS toLower(kw))\
  AND any(kw IN [\
    'neurodegeneration', 'alzheimer', 'parkinson', 'dementia', 'als', 'ms',\
    'tau', 'amyloid beta', 'alpha-synuclein', 'tdp-43', 'mapt', 'app',\
    'apoe', 'snca', 'grn', 'c9orf72', 'psen1', 'psen2',\
    'microglia', 'astrocyte', 'neuron', 'oligodendrocyte',\
    'dopamine', 'glutamate', 'acetylcholine', 'memantine', 'lithium',\
    'neuroinflammation', 'oxidative stress', 'synaptic loss', 'protein aggregation',\
    'memory loss', 'cognitive decline', 'tremor', 'bradykinesia',\
    'hippocampus', 'cortex', 'substantia nigra', 'brainstem'\
] WHERE toLower(end.name) CONTAINS toLower(kw))\
// Path constraints\
MATCH path = allShortestPaths((start)-[*..5]-(end))\
// Optional: enforce that intermediate nodes belong to biologically relevant types\
WHERE ALL(n IN nodes(path)[1..-1] \
  WHERE n:Gene OR n:Protein OR n:Chemical OR n:BiologicalProcess OR n:Cell OR n:Phenotype OR n:AnatomicalStructure)\
RETURN path\
LIMIT 20\
\
//Advanced\
\
// Find direct relationships between COVID-related and neurodegeneration-related entities\
MATCH (covid)-[r]->(neuro)\
WHERE any(kw IN ['covid', 'coronavirus', 'sars-cov-2', 'spike protein', 'ace2', 'cytokine storm', 'viral entry', 'interferon', 'il6', 'tnf', 'hypoxia'] \
    WHERE toLower(covid.name) CONTAINS toLower(kw))\
AND any(kw IN ['neurodegeneration', 'alzheimer', 'parkinson', 'dementia', 'tau', 'amyloid', 'alpha-synuclein', 'microglia', 'astrocyte', 'neuroinflammation', 'cognitive decline', 'memory loss'] \
    WHERE toLower(neuro.name) CONTAINS toLower(kw))\
RETURN \
    labels(covid) AS covid_type,\
    covid.name AS covid_entity,\
    type(r) AS relationship,\
    r.source AS source,\
    labels(neuro) AS neuro_type,\
    neuro.name AS neuro_entity,\
    count(*) AS frequency\
ORDER BY frequency DESC, source\
\
\
// Find intermediate mechanisms connecting COVID to neurodegeneration\
MATCH (covid)-[r1]->(bridge)-[r2]->(neuro)\
WHERE any(kw IN ['covid', 'coronavirus', 'sars-cov-2', 'spike protein', 'ace2'] \
    WHERE toLower(covid.name) CONTAINS toLower(kw))\
AND any(kw IN ['neurodegeneration', 'alzheimer', 'parkinson', 'tau', 'amyloid', 'microglia', 'neuroinflammation'] \
    WHERE toLower(neuro.name) CONTAINS toLower(kw))\
AND bridge:BiologicalProcess OR bridge:Chemical OR bridge:Gene OR bridge:Protein\
RETURN \
    covid.name AS covid_start,\
    type(r1) AS first_relationship,\
    r1.source AS source1,\
    labels(bridge) AS bridge_type,\
    bridge.name AS bridging_mechanism,\
    type(r2) AS second_relationship,\
    r2.source AS source2,\
    neuro.name AS neuro_end,\
    count(*) AS pathway_count\
ORDER BY pathway_count DESC\
LIMIT 20\
\
\
\
\
\
\
}