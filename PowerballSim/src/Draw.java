import java.util.Arrays;

public class Draw {
    private int[] commonNums;
    private int specialNum;

    public Draw(int[] common, int special) {
        Arrays.sort(common);
        commonNums = common;
        specialNum = special;
    }

    public int[] getCommon() {return commonNums;}

    public int getSpecialNum() {return specialNum;}

    @Override
    public String toString() {
        String str = "";

        for(int num : commonNums)
            str += ("["+num+"] ");

        str += "| ["+specialNum+"]";
        return str;
    }
}
