=============================================================
  DoQA: DOmain  specific  FAQs  via  conversational QA v2.1
=============================================================

DoQA is a dataset for accessing Domain Specific FAQs via
conversational QA that contains 2,437 information-seeking
question/answer dialogues (10,917 questions in total) on three
different domains: cooking, travel and movies. Note that we include in
the generic concept of FAQs also Community Question Answering sites,
as well as corporate information in intranets which is maintained in
textual form similar to FAQs, often referred to as internal "knowledge
bases".

These dialogues are created by crowd workers that play the following
two roles: the user who asks questions about a given topic posted in
Stack Exchange (https://stackexchange.com/), and the domain expert who
replies to the questions by selecting a short span of text from the
long textual reply in the original post. The expert can rephrase the
selected span, in order to make it look more natural. The dataset
covers unanswerable questions and some relevant dialogue acts.

DoQA enables the development and evaluation of conversational QA
systems that help users access the knowledge buried in domain specific
FAQs.



Dataset structure
-------------------

The dataset contains files for three different domains: cooking,
travel and movies. The cooking dataset is divided into three JSON
files in order to use the first one for the training process (train),
the second one for the tuning process (dev) and the last one for
testing (test). For the other two domains (travel and movies) only
test file is provided.

In DoQA we include a ranking of multiple documents that
can be relevant to answer a query and hence it may be used
to assess the ability of conversational QA systems to perform
a dialog when the exact passage with the correct answers are
not known.

Additionally, for the cooking domain we include a ranking of multiple
documents that can be relevant to answer a question and hence it may
be used to assess the ability of conversational QA systems to perform
a dialogue when the exact passage with the correct answers are not
nown. In the 'IR_scenario' folder we added two rank files for both
development and test data, which contain a ranking list of all
relevant answer passages for each dialogue (given the first question
in the dialogue), following the next two different strategies:

- question retrieval: where relevant or similar questions are searched
  (and thus, the answer for this relevant question is taken as a
  relevant answer); we indexed the original topics posted in the
  forum.

- answer retrieval: where relevant answers are searched directly among
  existing answers; we indexed the answer passages for each post in
  the forum.



The directories are distributed in the following way:

---doqa_dataset
    ---doqa-cooking-train-v2.1.json		Train split of the cooking dataset in json format

    ---doqa-cooking-dev-v2.1.json		Dev split of the cooking dataset in json format

    ---doqa-cooking-test-v2.1.json		Test split of the cooking dataset in json format

    ---doqa-travel-test-v2.1.json		Test split of the travel dataset in json format

    ---doqa-movies-test-v2.1.json		Test split of the movies dataset in json format


---ir_scenario
    ---doqa-cooking-dev-documents-v2.1.tsv	Documents and documents' ids of the dev split.
						File format (tsv):
						    document_id		 document

    ---doqa-cooking-dev-ir-answer-v2.1.tsv	Rank of all answer passages for each dialogue given by the 
						IR system in the dev file when the answers are
						indexed. File format (tsv):
						    dialogue_id	  	 document_id	 rank	score	 

    ---doqa-cooking-dev-ir-question-v2.1.tsv	Rank of all answer passages for each dialogue by the
						IR system in the dev file when the questions are
						indexed. File format (tsv):
						    dialogue_id	  	 document_id	 rank	score

    ---doqa-cooking-dev-relevants-v2.1.tsv	File containing the correct documents for each
						dialogue in the dev split. File format (tsv):
						    dialogue_id	     	 document_id

    ---doqa-cooking-test-documents-v2.1.tsv	Documents and documents' ids of the test split:
						File format (tsv):
						    document_id		 document

    ---doqa-cooking-test-ir-answer-v2.1.tsv	Rank of all answers passages for each dialogue given by the
						IR system in the dev file when the answers are
						indexed. File format (tsv):
						    dialogue_id	  	 document_id	 rank	score

    ---doqa-cooking-test-ir-question-v2.1.tsv	Rank of all answer passages for each dialogue given by the
						IR system in the dev file when the questions are
						indexed. File format (tsv):
						    dialogue_id	  	 document_id	 rank	score

    ---doqa-cooking-test-relevants-v2.1.tsv 	File containing the correct documents for each
						dialogue in the test split. File format(tsv):
						    dialogue_id	     	 document_id



You can find more information about the dataset in this publication:
  J.A. Campos, A. Otegi, A. Soroa, J. Deriu, M. Cieliebak,
  E. Agirre. DoQA - Accessing Domain-Specific FAQs via Conversational
  QA. Proceedings of the 58th Annual Meeting of the Association for
  Computational Linguistics. 2020



Authors
-----------
Jon Ander Campos (1), Arantxa Otegi (1), Aitor Soroa (1), Jan Deriu (2),
Mark Cieliebak (2), Eneko Agirre (1)

Affiliation of the authors: 
 (1) University of the Basque Country (UPV/EHU)
 (2) Zurich University of Applied Sciences (ZHAW)



Licensing
-------------
Copyright (C) by Ixa Taldea, University of the Basque Country UPV/EHU.
This dataset is licensed under the Creative Commons
Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0).
To view a copy of this license, visit
https://creativecommons.org/licenses/by-sa/4.0/legalcode.



Acknowledgements
-------------------
If you are using this dataset in your work, please cite this
publication:
  J.A. Campos, A. Otegi, A. Soroa, J. Deriu, M. Cieliebak,
  E. Agirre. DoQA - Accessing Domain-Specific FAQs via Conversational
  QA. Proceedings of the 58th Annual Meeting of the Association for
  Computational Linguistics. 2020



Contact information
-----------------------
Jon Ander Campos - jonander.campos@ehu.eus
