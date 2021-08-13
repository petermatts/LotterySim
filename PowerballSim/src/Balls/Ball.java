package Balls;

public class Ball {
    private final int number;
    private int frequency;
    private int daysAgo;

    public Ball(int number, int frequency, int daysAgo) {
        this.number = number;
        this.frequency = frequency;
        this.daysAgo = daysAgo;
    }

    public int getNumber() {return number;}

    public int getFrequency() {return frequency;}

    public int getDaysAgo() {return daysAgo;}
}
