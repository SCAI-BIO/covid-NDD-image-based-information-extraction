{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 // Node schema\
CALL db.schema.nodeTypeProperties()\
YIELD nodeLabels, propertyName, propertyTypes\
RETURN nodeLabels, propertyName, propertyTypes\
ORDER BY nodeLabels, propertyName;\
\
// Relationship schema\
CALL apoc.meta.relTypeProperties()\
YIELD relType, sourceNodeLabels, targetNodeLabels, propertyName, propertyTypes\
RETURN relType AS relationshipType,\
       sourceNodeLabels,\
       targetNodeLabels,\
       propertyName,\
       propertyTypes\
ORDER BY relationshipType, propertyName;\
}