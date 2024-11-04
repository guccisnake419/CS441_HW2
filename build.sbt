ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "Exercises441"
  )

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.3.4"
// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-mapreduce-client-core
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.4"
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.4"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1"
libraryDependencies += "org.nd4j" % "nd4j-native" % "1.0.0-M2.1"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.19"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % "test"
libraryDependencies += "org.scalatest" %% "scalatest-funsuite" % "3.2.19" % "test"
libraryDependencies += "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-M2.1"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.19"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % "test"
libraryDependencies += "org.scalatest" %% "scalatest-funsuite" % "3.2.19" % "test"
libraryDependencies += "org.mockito" %% "mockito-scala" % "1.16.46" % Test