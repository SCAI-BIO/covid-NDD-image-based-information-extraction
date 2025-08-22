{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // Projection for CBM source\
CALL gds.graph.drop('cbm-Graph', false);\
\
CALL gds.graph.project.cypher(\
  'cbm-Graph',\
  'MATCH (s)-[r]->(t) WHERE r.source = "CBM" RETURN DISTINCT id(s) AS id\
   UNION\
   MATCH (s)-[r]->(t) WHERE r.source = "CBM" RETURN DISTINCT id(t) AS id',\
  'MATCH (s)-[r]->(t) WHERE r.source = "CBM"\
   RETURN id(s) AS source, id(t) AS target, "UNDIRECTED" AS orientation'\
);\
\
// Community detection\
CALL gds.louvain.stream('cbm-Graph')\
YIELD nodeId, communityId\
RETURN gds.util.asNode(nodeId) AS node, communityId\
ORDER BY communityId\
LIMIT 20;\
\
}