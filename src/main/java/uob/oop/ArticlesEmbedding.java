package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if (!processedText.isEmpty()) return processedText;
        String news = super.getNewsContent();
        news = textCleaning(news);
        news = NLP(news);
        processedText = removeStopWords(news,Toolkit.STOPWORDS);
        return processedText.trim();
    }

    public static String NLP(String news)
    {
        StringBuilder sb = new StringBuilder();
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = pipeline.processToCoreDocument(news);
        for (CoreLabel tok : document.tokens())
        {
            sb.append(tok.lemma()).append(" ");
        }
        return sb.toString().trim().toLowerCase();
    }

    public static String removeStopWords(String _content, String[] _stopWords) {
        StringBuilder mySB = new StringBuilder();
        String[] wordsList = _content.split(" ");
        for (String word : wordsList) {
            if (notContains(_stopWords, word)) {
                mySB.append(word).append(" ");
            }
        }
        return mySB.toString().trim();
    }

    private static boolean notContains(String[] _arrayTarget, String _searchValue) {
        for (String element : _arrayTarget) {
            if (_searchValue.equals(element)) {
                return false;
            }
        }
        return true;
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        if(intSize==-1)
            throw new InvalidSizeException("Invalid Size");

        if(processedText.isEmpty())
            throw new InvalidTextException("Invalid text");

        if(!newsEmbedding.isEmpty())
            return Nd4j.vstack(newsEmbedding.mean(1));

        newsEmbedding = calcEMb(Nd4j.create(intSize, Toolkit.listVectors.get(0).length));
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    public INDArray calcEMb(INDArray newsEmbedding)
    {
        int pointer = 0;
        String[] words = processedText.split(" ");
        for (int i = 0; i < words.length; i++)
        {
            if (pointer >= intSize)
                break;
            for (int j = 0; j < AdvancedNewsClassifier.listGlove.size(); j++) {
                Glove g = AdvancedNewsClassifier.listGlove.get(j);
                if (g.getVocabulary().equals(words[i])) {
                    INDArray emb = Nd4j.create(g.getVector().getAllElements());
                    newsEmbedding.putRow(pointer, emb);
                    pointer++;
                }
            }
        }
        return newsEmbedding;
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
