package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        String[] stop = Toolkit.STOPWORDS;
        outerloop:
        for (int i = 0; i < Toolkit.listVocabulary.size(); i++)
        {
            String word = Toolkit.listVocabulary.get(i);
            for (int j = 0; j < stop.length; j++)
            {
                if(stop[j].equals(word))
                    continue outerloop;
            }
            listResult.add(new Glove(word, new Vector(Toolkit.listVectors.get(i))));
        }
        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        List<Integer> docLen = new ArrayList<>();
        for (int i = 0; i < _listEmbedding.size(); i++)
        {
            ArticlesEmbedding embedding = _listEmbedding.get(i);
            int length = 0;
            String[] words = embedding.getNewsContent().split(" ");
            for (int j = 0; j < words.length; j++)
            {
                String word = words[j];
                boolean flag = false;
                for (int k = 0; k < Toolkit.listVocabulary.size(); k++)
                {
                    if (Toolkit.listVocabulary.get(k).equals(word.trim())) {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    length++;
            }
            docLen.add(length);
        }

        docLen = sorting(docLen);
        intMedian = calcMedian(docLen.size(), docLen);

        return intMedian;
    }

    public int calcMedian(int size,List<Integer> docLen)
    {
        int intMedian =-1;
        if (size % 2 == 0)
        {
            int mid1 = docLen.get((size / 2) + 1);
            int mid2 = docLen.get(size / 2);
            intMedian= (mid1 + mid2) / 2;
        }
        else
        {
            intMedian= docLen.get((size+1) / 2);
        }
        return intMedian;
    }

    public List<Integer> sorting(List<Integer> docLen)
    {
        for (int i = 0; i < docLen.size(); i++)
        {
            for (int j = i + 1; j < docLen.size(); j++)
            {
                if (docLen.get(i) > docLen.get(j))
                {
                    int temp = docLen.get(i);
                    docLen.set(i, docLen.get(j));
                    docLen.set(j, temp);
                }
            }
        }
        return docLen;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (int i = 0; i < listEmbedding.size(); i++)
        {
            boolean done = false;
            while (!done)
            {
                ArticlesEmbedding ae = listEmbedding.get(i);
                try
                {
                    ae.getEmbedding();
                    done = true;
                }
                catch (InvalidSizeException e)
                {
                    ae.setEmbeddingSize(embeddingSize);
                }
                catch (InvalidTextException e)
                {
                    ae.getNewsContent();
                }
                catch (Exception e)
                {
                    System.err.println("Some error occurred " + e.getMessage());
                }
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks
        int pointer=0;
        for (int i=0; i<listEmbedding.size(); i++)
        {
            ArticlesEmbedding ae = listEmbedding.get(i);
            if(ae.getNewsType().equals(NewsArticles.DataType.Training))
            {
                try
                {
                    inputNDArray = ae.getEmbedding();
                    outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                    int[] a = new int[_numberOfClasses];
                    a[Integer.parseInt(ae.getNewsLabel()) - 1] = 1;
                    for (int j = 0; j < a.length; j++)
                        outputNDArray.putScalar(j, a[j]);
                }

                catch (Exception e)
                {
                    System.err.println("error occured "+e.getMessage());
                }
                DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(pointer,myDataSet);
                pointer++;
            }
        }

        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for(int i=0; i<_listEmbedding.size() ; i++)
        {
            ArticlesEmbedding ae = _listEmbedding.get(i);
            if(ae.getNewsType().equals(NewsArticles.DataType.Testing)) {
                int[] arr = myNeuralNetwork.predict(ae.getEmbedding());
                for (int j = 0; j < arr.length; j++) {
                    listResult.add(arr[j]);
                }
                ae.setNewsLabel(Integer.toString(arr[0]));
            }
        }
        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        int grpNo = maxSize();
        List<String>[] gList = new List[grpNo];

        for (int i = 0; i < grpNo; i++)
            gList[i] = new ArrayList<>();

        boolean[] grpIs = new boolean[grpNo];
        grpIs = exist(grpIs, gList);
        print(grpNo, grpIs, gList);
    }

    public void print(int grpNo,boolean[] grpIs, List<String>[] gList)
    {
        for(int i=0; i<grpNo ; i++)
        {
            if(grpIs[i])
            {
                System.out.println("Group "+(i+1));
                for (int j = 0; j < gList[i].size(); j++)
                    System.out.println(gList[i].get(j));
            }
        }
    }

    public boolean[] exist(boolean[] grpIs, List<String>[] gList)
    {
        for(int i=0; i<listEmbedding.size(); i++)
        {
            ArticlesEmbedding ae = listEmbedding.get(i);
            if(ae.getNewsType().equals(NewsArticles.DataType.Testing))
            {
                String label = ae.getNewsLabel();
                String title= ae.getNewsTitle();
                gList[Integer.parseInt(label)].add(title);
                grpIs[Integer.parseInt(label)] = true;
            }
        }
        return grpIs;
    }

    public int maxSize()
    {
        List<String> el = new ArrayList<>();
        for(int i=0; i<listEmbedding.size(); i++)
        {
            ArticlesEmbedding ae = listEmbedding.get(i);
            if(ae.getNewsType().equals(NewsArticles.DataType.Testing))
            {
                if(!el.contains(ae.getNewsLabel()))
                    el.add(ae.getNewsLabel());
            }
        }
        return el.size();
    }
}
