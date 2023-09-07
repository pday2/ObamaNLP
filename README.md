# Shaping a Leader's Words
## An analysis of Barack Obama’s speeches through natural language processing

This is the repository for our research for our Master's Thesis at KU Leuven. 

##Kylan Young and Peter Day

* encodingPCA2.ipynb - Spacy encodings, PCA analysis and biplots
* posTagging.ipynb - Stanza part of Speech tagging and proportion calculations
* nrcEmotionAnalysis.ipynb - NRCLex emotion analysis
* stanzaDependency.ipynb - Sentence dependency parsing and depth calculations
* sentenceLength.ipynb - Word, syllable counts
* readabilityScores.ipynb - Word, Syllable and Readability score calculations
* textPCASimilarity.ipynb - PCA comparison of Obama, NYT and WSJ using calculated features
* varianceObaGWB.ipynb - Investigating variance differences between Obama and Bush
* correspondenceAnalysis.ipynb - compare Obama and Bush
* correspondenceAnalysis_amrhet.ipynb - includes Selma speech
* parallelism.ipynb - Parallel grammar construct detector and counter
* obama_vs_gwb.ipynb - Comparing Obama to Bush

------------------------------------------------------------------------------

# Data Directories

* Data - 455 [Obama Speeches from American Rhetoric](https://www.americanrhetoric.com/barackobamaspeeches.htm) in txt and csv
* DataUCSB - 129 [Obama speeches from U Cal Santa Barbara](https://www.presidency.ucsb.edu/documents/barack-obama-event-timeline) in txt and csv
* DataWiki - [Wikipedia timelines of Obama's presidential years](https://en.wikipedia.org/wiki/Timeline_of_the_Barack_Obama_presidency) csv
* GWB - 82 [George W Bush speeches from U Cal Santa Barbara](https://www.presidency.ucsb.edu/documents/george-w-bush-event-timeline)
* NYT - 101 news articles from The New York Times
* OtherSources - 10 articles from other news sources
* speeches - 101 Obama speeches for comparison with NYT and WSJ
* Top10 - 10 of Obama's most popular speeches
* word_lists - Various lists of words, stopwords, Dale-Chall easy word list
* WSJ - 101 news articles from The Wall Street Journal

------------------------------------------------------------------------------

> He took a swig of unsweetened iced tea and sat forward. “Let me put it another way. You listen to Miles Davis?”

> I worried this was a test to see if I’d followed John Coltrane down the rabbit hole. I said yes.

> “You know what they say about Miles Davis?”

> I did not.

> “It’s the notes you don’t play. It’s the silences. That’s what made him so good. I need a speech with some pauses, and some quiet moments, because they say something too. You feel me?”

> By that point, I did. I knew exactly what he was talking about. What brief pride I’d felt in a speech that was in the best shape it had ever been a week ahead of time was quickly replaced by regret—regret that I’d been so consumed with making sure everything was in there that it made him complain that everything was in there.

> “Good,” he finished. “Like I said, we’re in great shape. I don’t want you to do any work tonight. I want you to go home, pour yourself a drink, and listen to some Miles Davis. And tomorrow, take another swing at it.”

> He pointed his fork at me. “Find me some silences.”

> Keenan, Cody. Grace (p. 156). 
