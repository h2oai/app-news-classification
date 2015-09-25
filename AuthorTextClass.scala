/**
 *  * Craigslist example
 *
 * It predicts job category based on job description (called "job title").
 *
 * Launch following commands:
 *    export MASTER="local-cluster[3,2,4096]"
 *   bin/sparkling-shell -i examples/scripts/craigslistJobTitles.script.scala
 *
 * When running using spark shell or using scala rest API:
 *    SQLContext is available as sqlContext
 *    SparkContext is available as sc
 */
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.DataFrame


def isHeader(line: String) = line.contains("category")
// Load and split data based on ","
val data = sc.textFile("/Users/avniwadhwa/Desktop/MeetUp/ExperimentalNewOtherTechCrunch.csv").filter(x => !isHeader(x)).map(d => d.split(',')).filter(row => row.length == 5)


// Extract job category from job description
val articleAuthor = data.map(l => l(3))
val articleTags = data.map(l => l(2))
val articleText = data.map(l => l(4))
// Count of different job categories
val labelCounts = articleText.map(n => (n, 1)).reduceByKey(_+_).collect.mkString("\n")

/* Should return:
(education,2438)
(administrative,2500)
(labor,2500)
(accounting,1593)
(customerservice,2319)
(foodbeverage,2495)
*/

// All strings which are not useful for text-mining
val stopwords = Set("ax","i","you","edu","s","t","m","subject","can","lines","re","what"
    ,"there","all","we","one","the","a","an","of","or","in","for","by","on", "also", "now"
    ,"but", "is", "in","a","not","with", "as", "was", "if","they", "are", "this", "and", "it", "have"
    , "from", "at", "my","be","by","not", "that", "to","from","com","org","like","likes","so"
    , "you", "your", "they", "them", "our", "fit", "week", "how", "why", "what", "when", "lets"
    , "look", "could", "here", "there", "after", "before", "will", "would", "should", "open" 
    , "has", "he", "she", "his", "her", "some", "no", "us", "more", "just", "recent", "even", "non", "many")

// Compute rare words
val rareWords = articleText.flatMap(t => t.split("""\W+""").map(_.toLowerCase)).filter(word => """[^0-9]*""".r.pattern.matcher(word).matches).
  map(w => (w, 1)).reduceByKey(_+_).
  filter { case (k, v) => v < 5  || v > 40 }.map { case (k, v) => k }.
  collect.
  toSet

val topTenAuthors = articleAuthor.map(a => (a, 1)).reduceByKey(_+_).sortBy(- _._2).map(v => v._1).take(100)

// Define tokenizer function
def token(line: String): Seq[String] = {
    //get rid of nonWords such as puncutation as opposed to splitting by just " "
    val result = line.split("""\W+""") 
    .map(_.toLowerCase)

    //remove mix of words+numbers
    .filter(word => """[^0-9]*""".r.pattern.matcher(word).matches) 

    //remove stopwords defined above (you can add to this list if you want)
    .filterNot(word => stopwords.contains(word)) 

    //leave only words greater than 1 characters. 
    //this deletes A LOT of words but useful to reduce our feature-set
    .filter(word => word.size >= 2) 

    //remove rare occurences of words
    .filterNot(word => rareWords.contains(word)) 

    result
}


// YYYY
val words = data.map(d => (d(2), token(d(4)).toSeq, d(3))).filter(s => s._2.length > 0)

val YYYlabels = words.map(v => v._1)
val YYYtext = words.map(v => v._2)
val YYYauthor = words.map(v => v._3)

//-----

// Sanity Check
println(articleText.flatMap(lines => token(lines)).distinct.count) 
println(articleAuthor.flatMap(lines => token(lines)).distinct.count) 

// Make some helper functions
def sumArray (m: Array[Double], n: Array[Double]): Array[Double] = {
  for (i <- 0 until m.length) {m(i) += n(i)}
  return m
}

def divArray (m: Array[Double], divisor: Double) : Array[Double] = {
  for (i <- 0 until m.length) {m(i) /= divisor}
  return m
}

def wordToVector (w:String, m: Word2VecModel): Vector = {
  try {
    return m.transform(w)
  } catch {
    case e: Exception => return Vectors.zeros(100)
  }  
}

//
// Word2Vec Model
// 

val word2vec = new Word2Vec()
val model4Text = word2vec.fit(YYYtext)



// Sanity Check
//model.findSynonyms("start", 5).foreach(println)

val textVectors = YYYtext.map(x => new DenseVector(
    divArray(x.map(m => wordToVector(m, model4Text).toArray).
            reduceLeft(sumArray),x.length)).asInstanceOf[Vector])


//model1.findSynonyms("start", 5).foreach(println)




//val title_pairs = words.map(x => (x,new DenseVector(
//    divArray(x.map(m => wordToVector(m, model).toArray).
//            reduceLeft(sumArray),x.length)).asInstanceOf[Vector]))

// Create H2OFrame
import org.apache.spark.mllib
//case class TECHCRUNCH(target: String, text: mllib.linalg.Vector, title: mllib.linalg.Vector)
case class TECHCRUNCH(target: String, author:String, text: mllib.linalg.Vector)

import org.apache.spark.h2o._
val h2oContext = new H2OContext(sc).start()

//val resultRDD: DataFrame = YYYlabels.zip(textVectors).zip(titleVectors).map(v => TECHCRUNCH(v._1._1, v._1._2, v._2)).toDF
val resultRDD: DataFrame = YYYlabels.zip(YYYauthor).zip(textVectors).map(v => {
  val author = v._1._2
  val a = if (topTenAuthors.contains(author)) author else "nothing"
  TECHCRUNCH(v._1._1, a, v._2)}
  ).toDF


val table:H2OFrame = h2oContext.asH2OFrame(resultRDD)

// OPEN FLOW UI
h2oContext.openFlow

/// ====
// Input data
val testAuthor = "Michael Seo"
val testText = "The Infinity Cell is a kinetic charger for the iPhone that uses your bodyÕs movement to generate electricity. The current prototype for the Infinity Cell is a crude 3D printed rectangle roughly the size of a pack of cigarettes linked up to the iPhone with a cable. The plan is to create a more streamlined version during the productÕs Kickstarter campaign."
// Tokenize the text
val testTextTokens = token(testText).toSeq
// Use word2vec model to transform tokenized text to feature vector
val testTextVector = new DenseVector(divArray(testTextTokens.map(m => wordToVector(m, model4Text).toArray).reduceLeft(sumArray), testTextTokens.length))

// Transforming input data into H2OFrame
val testRow = TECHCRUNCH(null, testAuthor, testTextVector)
val testRowRdd = sc.parallelize(Seq(testRow)).toDF
val testRowHf: H2OFrame = h2oContext.asH2OFrame(testRowRdd)
// Removing 1st column, since there is a bug in H2O scoring
testRowHf.remove(0)
testRowHf.update(null)



