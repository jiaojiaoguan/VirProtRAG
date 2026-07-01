#!/bin/bash
#SBATCH -e VirProtRAG.%j.e
#SBATCH -o VirProtRAG.%j.o
#SBATCH -J VirProtRAG
#SBATCH --export=all
#SBATCH -p cpu1
#SBATCH --cpus-per-task=8




# .env 中已设 MEDCPT_FAISS_INDEX_PATH 和 MEDCPT_PMIDS_PATH，无需额外指定
# virprotrag --phase medcpt --input test_bm25.json --output test_medcpt.json --verbose

virprotrag batch --phase medcpt --input batch_bm25/ --output batch_medcpt/