package uob.oop;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 marks
        listVectors = new ArrayList<>();
        listVocabulary = new ArrayList<>();
        try
        {
            String line;
            myReader = new BufferedReader(new FileReader(Toolkit.getFileFromResource(FILENAME_GLOVE)));
            while((line=myReader.readLine()) != null)
            {
                String[] words = line.split(",");
                String word = words[0];
                listVocabulary.add(word);

                double[] vec = new double[words.length-1];
                for (int i = 0; i < vec.length ; i++)
                    vec[i] = Double.parseDouble(words[i+1]);

                listVectors.add(vec);
            }
        }
        catch(URISyntaxException e)
        {
            System.err.println("URI syntax Error "+ e.getMessage());
        }
        catch(FileNotFoundException e)
        {
            System.err.println("File not exist "+e.getMessage());
            throw e;
        }
        catch (Exception e)
        {
            System.err.println("Error "+e.getMessage());
        }
        finally {
            if(myReader != null)
            {
                try
                {
                    myReader.close();
                }
                catch(IOException e)
                {
                    System.err.println("Error closing BufferedReader: " + e.getMessage());
                }
                catch (Exception e)
                {
                    System.err.println("Error "+e.getMessage());
                }
            }
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public File[] sorting(File[] allFiles)
    {
        int n = allFiles.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (allFiles[j].getName().compareTo(allFiles[j + 1].getName()) > 0) {
                    File temp = allFiles[j];
                    allFiles[j] = allFiles[j + 1];
                    allFiles[j + 1] = temp;
                }
            }
        }
        return allFiles;
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks

        try
        {
            File newsFilePath = getFileFromResource("News");
            File[] allFiles = newsFilePath.listFiles();
            if(allFiles != null) {
                allFiles = sorting(allFiles);

                for (int i = 0; i < allFiles.length; i++) {
                    if (allFiles[i].isFile() && allFiles[i].getName().endsWith(".htm")) {
                        String code = Files.readString(allFiles[i].toPath());
                        listNews.add(new NewsArticles(HtmlParser.getNewsTitle(code), HtmlParser.getNewsContent(code), HtmlParser.getDataType(code), HtmlParser.getLabel(code)));
                    }
                }
            }
            else
                System.err.println("No file exists in news folder");
        }
        catch (URISyntaxException e)
        {
            System.err.println("URISyntax Exception "+e.getMessage());
        }
        catch(Exception e)
        {
            System.err.println("Something went wrong "+e.getMessage());
        }
        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
