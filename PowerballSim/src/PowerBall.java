import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.jsoup.nodes.Element;

public class PowerBall {
    private Map<Integer, Integer> WhiteBalls;
    private Map<Integer, Integer> PowerBalls;
    private int totalDraws;
    private List<Integer> WBs;
    private List<Integer> PBs;
    private Draw lastDraw;

    public PowerBall(List<Element> WBs, List<Element> PBs) throws Exception {
        WhiteBalls = new HashMap<Integer, Integer>();
        PowerBalls = new HashMap<Integer, Integer>();
        this.WBs = new ArrayList<Integer>();
        this.PBs = new ArrayList<Integer>();
        lastDraw = null;

        int commonTotal = 0;
        int powerTotal = 0;

        for(Element white : WBs) {
            int num = Integer.parseInt(white.attr("data-num"));
            int freq = Integer.parseInt(white.attr("data-freq"));
            commonTotal += freq;
            
            for(int i = 0; i < freq; i++)
                this.WBs.add(num);

            WhiteBalls.put(num, freq);
        }

        for(Element power : PBs) {
            int num = Integer.parseInt(power.attr("data-num"));
            int freq = Integer.parseInt(power.attr("data-freq"));
            powerTotal += freq;

            for(int i = 0; i < freq; i++)
                this.PBs.add(num);

            PowerBalls.put(num, freq);
        }

        if(commonTotal/5 == powerTotal)
            totalDraws = powerTotal;
        else
            throw new Exception();
    }

    public Draw draw() {
        Random random = new Random();
        Set<Integer> common = new HashSet<Integer>();
        while(common.size() != 5) {
            int randindex = random.nextInt(WBs.size());
            common.add(WBs.get(randindex));
        }

        Object[] array = common.toArray();
        int[] arr = new int[array.length];
        for(int i = 0; i < common.size(); i++)
            arr[i] = (Integer) array[i];

        Arrays.sort(arr);
        lastDraw = new Draw(arr, PBs.get(random.nextInt(PBs.size())));
        return lastDraw;
    }

    public int numCommonElements(int[] a, int[] b) {
        Integer[] A = Arrays.stream(a).boxed().toArray(Integer[]::new);
        Integer[] B = Arrays.stream(b).boxed().toArray(Integer[]::new);
        Set<Integer> setA = new HashSet<Integer>(Arrays.asList(A));
        Set<Integer> setB = new HashSet<Integer>(Arrays.asList(B));
        setA.retainAll(setB);
        return setA.size();
    }

    private String checkDraw(Draw draw) {
        final boolean pb = lastDraw.getSpecialNum() == draw.getSpecialNum();
        final int matches = numCommonElements(lastDraw.getCommon(), draw.getCommon());

        if(matches == 5 && pb)
            return "JACKPOT\n";
        else if(matches == 5 && !pb)
            return "5\n";
        else if(matches == 4 && pb)
            return "4 + POWERBALL\n";
        else if(matches == 4 && !pb)
            return "4\n";
        else if(matches == 3 && pb)
            return "3 + POWERBALL\n";
        else if(matches == 3 && !pb)
            return "3\n";
        else if(matches == 2 && pb)
            return "2 + POWERBALL\n";
        else if(matches == 1 && pb)
            return "1 + POWERBALL\n";
        else if(pb)
            return "POWERBALL\n";
        else
            return "Lost\n";
    }

    public String checkTicket(Ticket ticket) {
        String str = "";
        String t = ticket.toString();
        String[] splitt = t.split("\n");
        Draw[] tDraws = ticket.getDraws();
        for(int i = 0; i < splitt.length; i++)
            str += splitt[i] + "\t" + checkDraw(tDraws[i]);

        return str;
    }

    @Override
    public String toString() {
        String str = "PowerBall\n";
        String breaks = "\n\n";

        str += "# of Draws: " + totalDraws + breaks;
        str += "Common Balls\n";
        str += WhiteBalls.toString();
        str += breaks;
        str += "Powerballs\n";
        str += PowerBalls.toString();
        str += breaks;

        return str;
    }
}
