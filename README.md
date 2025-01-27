# SaexyLLM

- fine tuning: trained on the crawled sachsen text with german-gpt2. For now only trained partial text and the generated text is a poem in hochdeutsch.

- sachsen_dictionary: use the same german-gpt2 model to find the perplexity of the text in sachsen dataset. Find the closest word in the hochdeutsch to sachsen dialect dictionary. Get a dictionary which can be used for this sachsen poem dataset. Last step replace all the confused words with the newly generated dictionary. (Problems: the hochdeustch to sachsen dialect dictionary is very limited; in the sachsen dataset, there are several words that are combined together which confuses the model)