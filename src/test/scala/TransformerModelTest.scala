import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.{be, noException}

class TransformerModelTest extends AnyFunSuite {


  test("loadPretrainedModel should load embeddings with non-empty results") {
    val samplePath = "src/main/resources/input/data.txt" // A path to a sample embeddings file for testing
    val embeddings = TransformerModel2.loadPretrainedModel(samplePath)
    assert(embeddings.nonEmpty, "Embeddings should not be empty")
    assert(embeddings.head._2.length > 0, "Each embedding should have a non-zero length vector")
  }

  test("createSlidingWindowsWithPositionalEmbedding should create DataSet with expected shapes") {
    val samplePath = "src/main/resources/input/data.txt" // A path to a sample embeddings file for testing
    val embeddings = TransformerModel2.loadPretrainedModel(samplePath)
    val windowSize = 4
    val datasets = TransformerModel2.createSlidingWindowsWithPositionalEmbedding(embeddings, windowSize)
    assert(datasets.nonEmpty, "Datasets should not be empty")
    assert(datasets.head.getFeatures.shape()(1) == windowSize, "Features should have the correct window size")
  }

  test("computePositionalEmbedding should return expected shape for positional encoding") {
    val windowSize = 5
    val dataLength = 10
    val positionalEncoding = TransformerModel2.computePositionalEmbedding(windowSize, dataLength)
    assert(positionalEncoding.shape()(0) == windowSize, "Positional encoding should have correct window size")
    assert(positionalEncoding.shape()(1) == dataLength, "Positional encoding should have correct embedding dimension")
  }

  test("An empty Set should have size 0") {
    assert(Set.empty.size == 0)
  }
  test("Invoking head on an empty Set should produce NoSuchElementException") {
    assertThrows[NoSuchElementException] {
      Set.empty.head
    }
  }

  // Stop the Spark context after tests

}