import org.apache.spark
import org.apache.spark.mllib.clustering.LocalLDAModel
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingLayer, LSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.{InvocationType, IterationListener}
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}

import java.util
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.LossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.shade.wstx.util.DataUtil.Integer

import java.lang.management.ManagementFactory
import java.time.LocalDateTime
import scala.io.Source
import scala.jdk.CollectionConverters.seqAsJavaListConverter

object TransformerModel2 {
  //the input data is what we used in hw1
  //read from hw1 output file...ideally, loadPretrainedModel
  //Save the model in MultiLayerNetwork
  //
  private val transformerLogger = LoggerFactory.getLogger("TranformerModel")

  def createSlidingWindowsWithPositionalEmbedding(tokens: Array[(String, Array[Double])], windowSize: Int): List[DataSet] = {
    tokens.sliding(windowSize *2).toList.map { window =>
      // Separate input window and target token
      val inputWindow = window.take(windowSize)
      val targetToken = window.takeRight(windowSize)

      // Stack embeddings for the input window
      val temp = inputWindow.map(_._2)
      val inputEmbeddings = Nd4j.create(temp).reshape(1,windowSize, temp.head.length)

      // Compute positional embeddings for the window size
      val positionalEmbeddings = computePositionalEmbedding(windowSize, inputEmbeddings.length/windowSize)
      val positionAwareEmbedding = inputEmbeddings.add(positionalEmbeddings)
      // Target embedding is the last element in the window
      val targetEmbedding = Nd4j.create(targetToken.map(_._2)).reshape(1, windowSize, temp.head.length)
      transformerLogger.debug(s"Position-aware embedding shape: ${positionAwareEmbedding.shape().mkString(", ")}")
      transformerLogger.debug(s"Target embedding shape: ${targetEmbedding.shape().mkString(", ")}")
      // Create and return the DataSet
      new DataSet(positionAwareEmbedding, targetEmbedding)
    }
  }




  def createSparkContext: SparkContext = {
    // Configure Spark for local or cluster mode
    val sparkConf = new SparkConf().setAppName("DL4J-LanguageModel-Spark").setMaster("local[*]") // For local testing, or use "yarn", "mesos", etc. in a cluster

    // Create Spark context
    val sc = new SparkContext(sparkConf)
    sc
  }

  // Compute sinusoidal positional embeddings for a given window size
   def computePositionalEmbedding(windowSize: Int, dataLength : Long) = {
    val embeddingDim = dataLength // Dimensionality of word embeddings

    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)
    for (pos <- 0 until windowSize) {
      var i = 0
      while (i < embeddingDim) {
        val angle = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        positionalEncoding.putScalar(Array[Int](pos, i), Math.sin(angle))
        positionalEncoding.putScalar(Array[Int](pos, i + 1), Math.cos(angle))

        i += 2
      }
    }
    positionalEncoding
  }

  def loadPretrainedModel(modelPath: String): Array[(String, Array[Double])] =  {
    //read hw1 outputs
    //separate by tokens and embeddings
    val source = Source.fromFile(modelPath)
    val embeddings = source.getLines().flatMap { line =>
      val parts = line.split("\t")
      if (parts.length >= 2) {
        val word = parts(0).trim
        val embeddingsStr = parts(1).replace("Average Embeddings: Array(", "").replace(")", "")
        val embeddingArray = embeddingsStr.split(", ").map(_.toDouble)
        Some(word -> embeddingArray)
      } else {
        None
      }
    }.toArray
    source.close()
    embeddings
  }

  def createRDDFromData(data: List[DataSet], sc: SparkContext): RDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    val rddData = sc.parallelize(data)
    rddData
  }



  def main(args: Array[String]): Unit = {
    val data = loadPretrainedModel(args(0))
    val vocab = data.map(_._1).distinct.zipWithIndex.toMap
    val sc = createSparkContext
    transformerLogger.info("Total executors: " + sc.getExecutorMemoryStatus.size)
    val slidingWindow= createSlidingWindowsWithPositionalEmbedding(data, 4)
    val slidingWindowDataSet = createRDDFromData(slidingWindow, sc)

    val conf = new NeuralNetConfiguration.Builder()
      .list()
      .layer(0, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(4)
        .nOut(4)
        .activation(Activation.IDENTITY)
        .build())
      .build()

    val  model = new MultiLayerNetwork(conf);
    model.init()

    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(1)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
      .batchSizePerWorker(1)
      .build()

//    // Create a SparkDl4jMultiLayer with the Spark context and model
    val sparkModel = new SparkDl4jMultiLayer(sc,model , trainingMaster)

    model.setListeners(new ScoreIterationListener(10))
    model.setIterationCount(10)

    val startTime = System.currentTimeMillis
    transformerLogger.info(s"Training started at ${LocalDateTime.now()}")
    // Set listeners to monitor the training progress
    transformerLogger.info(s"Set current listener to 10" + model.setListeners(new ScoreIterationListener(10)))

    // Log gradient stats every 1 iteration
    transformerLogger.info(s" Current Gradient stats: " + model.setListeners(new GradientStatsListener(1)))

    // Log partition
    transformerLogger.info(s"Number of Partition: ${slidingWindowDataSet.getNumPartitions}")

    // Log Current Learning Rate
    transformerLogger.info("Current Learning Rate : " + (model.getLearningRate(0)))

    // Log CPU usage
    transformerLogger.info(s" CPU Usage: ${ManagementFactory.getOperatingSystemMXBean.getSystemLoadAverage}")
    transformerLogger.info(s"JVM Memory Usage: Used: " +
      s"${Runtime.getRuntime.totalMemory - Runtime.getRuntime.freeMemory} bytes, " +
      s"Total: ${Runtime.getRuntime.totalMemory} bytes")

    sparkModel.fit(slidingWindowDataSet)

    // Log Training end time
    val endTime = System.currentTimeMillis
    transformerLogger.info(s"Training ended at ${LocalDateTime.now()}")

    // Log Epoch tme
    transformerLogger.info(s"Epoch time: " + (endTime - startTime) + " ms")
//    // Stop the Spark context after training
    sc.stop()

  }

}
