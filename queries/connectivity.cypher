{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // Nodes connected by source\
MATCH (a)-[r]->(b)\
WHERE r.source IS NOT NULL\
WITH r.source AS source, collect(DISTINCT a) + collect(DISTINCT b) AS all_nodes\
RETURN source, size(all_nodes) AS unique_nodes_count\
ORDER BY unique_nodes_count DESC;\
\
// Connected components per source (APOC + GDS)\
MATCH ()-[r]->()\
WITH DISTINCT r.source AS src\
WITH src, "graph_" + toString(src) AS graphName\
CALL apoc.cypher.run(\
  '\
  CALL gds.graph.project.cypher(\
    $graphName,\
    "MATCH (n) RETURN id(n) AS id",\
    "MATCH (n)-[r]->(m) WHERE r.source = \\'" + $source + "\\' RETURN id(n) AS source, id(m) AS target"\
  )\
  YIELD graphName AS gname1\
\
  CALL gds.wcc.stats(gname1)\
  YIELD componentCount\
\
  WITH componentCount, gname1, $source AS source\
  CALL gds.graph.drop(gname1) YIELD graphName AS droppedGraph\
\
  RETURN source, componentCount\
  ',\
  \{source: src, graphName: graphName\}\
) YIELD value\
RETURN value.source AS source, value.componentCount AS componentCount;\
}