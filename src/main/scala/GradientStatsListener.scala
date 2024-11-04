import org.apache.spark.ml.Model
import org.deeplearning4j.nn.api
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory

import java.time.LocalDateTime
import java.util

class GradientStatsListener(frequency: Int) extends TrainingListener {
  private val logger = LoggerFactory.getLogger(classOf[GradientStatsListener])
  private var iterationCount: Int = 0


  override def iterationDone(model: api.Model, iteration: Int, epoch: Int): Unit = ???

  override def onEpochStart(model: api.Model): Unit = {
    logger.info(s"Epoch started at ${LocalDateTime.now()}.")

  }

  override def onEpochEnd(model: api.Model): Unit = {
    logger.info(s"Epoch started at ${LocalDateTime.now()}.")
  }

  override def onForwardPass(model: api.Model, activations: util.List[INDArray]): Unit = {
    logger.info(s"Finished Forward pass ${LocalDateTime.now()}. Number of activations: ${activations.size()}")
  }

  override def onForwardPass(model: api.Model, activations: util.Map[String, INDArray]): Unit =  {
    logger.info(s"Finished Forward pass . Number of activations: ${activations.keySet()}")
  }


  override def onGradientCalculation(model: api.Model): Unit = {
    // Increment the iteration count manually
    iterationCount += 1

    // Log gradient stats at the specified frequency
    if (iterationCount % frequency == 0) {
      val gradients = model.gradient().gradientForVariable().values()
      var totalNorm = 0.0

      gradients.forEach { grad =>
        val norm = grad.norm2Number().doubleValue()
        totalNorm += norm
        logger.info(s"Gradient L2 Norm: $norm")
      }

      logger.info(s"Total Gradient L2 Norm: $totalNorm")
    }
  }

  override def onBackwardPass(model: api.Model): Unit = {
    logger.info("Finished Backward pass.")

  }
}
