import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

//import Balls.*;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class App {
    public static void main(String[] args) throws Exception {
        Document doc = Jsoup.connect("https://www.powerball.net/statistics").get();
        Elements elems = doc.getElementsByClass("freq-result js-stats-item"); //? use "js-stats" instead and work to data using children methods
        List<Element> list = new ArrayList<Element>();
        //int i = 0;
        for(Element elem : elems) {
            list.add(elem);
            // System.out.println(i +"\t"+ elem.attr("data-num") +"\t"+ elem.attr("data-freq") +"\t"+ elem.attr("data-ago"));
            //i++;
        }
        List<Element> WB = list.subList(678, 747); //WhiteBall Elements
        List<Element> PB = list.subList(747, 773); //Powerball Elements

        Scanner scan = new Scanner(System.in);
        PowerBall lottery = new PowerBall(WB, PB);

        int[] a = {23, 32, 45, 61, 69};
        int[] b = {9, 17, 37, 44, 54};
        Draw our1 = new Draw(a, 4);
        Draw our2 = new Draw(b, 14);
        Ticket t = new Ticket(our1, our2);

        System.out.println(lottery);
        while(true) {
            System.out.println("Would you like to draw (any key) or exit (n) ?");
            String input = scan.nextLine();
            if(input.equalsIgnoreCase("n"))
                break;
            else {
                System.out.println(lottery.draw());
                System.out.println(lottery.checkTicket(t));
                System.out.println();
            }
        }

        scan.close();
    }
}
