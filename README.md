# ece-5990-experiments


# ideas - 

- Adaptive query processing 
    - Depending upon the complexity of the query and the effectiveness of retrival, dynamically identify which LLM to use for optimizing the cost associated with generation 

    - Use a judge to estimate the cost associated with processing a query beforhand and use that to identify which LLM to use 

    - Process the same query using multiple LLMs and use the judge to identify which output was more relevant, use this context to improve the choice of which LLM to use 

- Adaptive RAG 
    - Multi step RAG 
    - Query Routing 
     -   eg. route customer query regarding a specific product to the subsection of a QA bank 
    - Adaptive retrieval 
        - Dynamically identify how much chunks to retrieve depending on the query complexity 
        - Confidence score will determine if more chunks are needed 

- Optimized RAG
    - Optimize the speed of RAG by retrieving chunks in parallel 
    - Since, Large data sets are often stored in distributed data lakes - RAG can borrow from distibuted data processing capabilities and accelerate the context retrival 