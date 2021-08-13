public class Ticket {
    private Draw[] draws;

    @SafeVarargs
    public Ticket(Draw ...d) throws IllegalArgumentException {
        if(d.length > 5)
            throw new IllegalArgumentException();

        draws = new Draw[5];
        for(int i = 0; i < d.length; i++)
            draws[i] = d[i];           
    }

    public Draw[] getDraws() {return draws;}

    @Override
    public String toString() {
        String str = "";

        // str += "Ticket " + this.hashCode(); //optional
        for(int i = 0; i < draws.length && draws[i]!=null; i++)
            str += draws[i].toString() + "\n";

        return str;
    }
}
