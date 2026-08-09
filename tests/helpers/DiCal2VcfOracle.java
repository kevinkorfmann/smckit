import edu.berkeley.diCal2.haplotype.HFSAXFullHaplotype;
import edu.berkeley.diCal2.haplotype.ReadSequences;
import edu.berkeley.diCal2.haplotype.SimpleFSARef;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Minimal direct oracle for the immutable diCal2 VCF reader. */
public final class DiCal2VcfOracle {
    private static String values(int[] input) {
        StringBuilder output = new StringBuilder();
        for (int index = 0; index < input.length; index++) {
            if (index > 0) output.append(',');
            output.append(input[index]);
        }
        return output.toString();
    }

    private static List<Boolean> mask(String encoded) {
        List<Boolean> output = new ArrayList<Boolean>();
        for (String value : encoded.split(",", -1)) {
            output.add(value.equals("1"));
        }
        return output;
    }

    public static void main(String[] args) {
        if (args.length != 6) {
            throw new IllegalArgumentException(
                "usage: VCF REFERENCE N_ALLELES INCLUDE_MASK ACCEPT_UNPHASED IGNORE_DUPLICATES"
            );
        }
        try {
            List<HFSAXFullHaplotype> haplotypes = ReadSequences.readVcf(
                new FileReader(args[0]),
                mask(args[3]),
                Integer.parseInt(args[2]),
                new char[] {'#'},
                Boolean.parseBoolean(args[4]),
                args[0],
                0,
                null,
                false,
                ".",
                args[1],
                Boolean.parseBoolean(args[5])
            );
            HFSAXFullHaplotype first = null;
            for (HFSAXFullHaplotype haplotype : haplotypes) {
                if (haplotype != null) {
                    first = haplotype;
                    break;
                }
            }
            if (first == null) throw new IllegalStateException("oracle selected no haplotypes");
            SimpleFSARef reference = (SimpleFSARef) first.getReference();
            int[] segregating = reference.getSegSites();
            Arrays.sort(segregating);
            System.out.println("SMCKIT_STATUS\tOK");
            System.out.println("SMCKIT_SEG\t" + values(segregating));
            System.out.println(
                "SMCKIT_REFERENCE\t" + values(SimpleFSARef.byteArrayToIntArray(reference.getAlleleConfig()))
            );
            int row = 0;
            for (HFSAXFullHaplotype haplotype : haplotypes) {
                if (haplotype == null) continue;
                System.out.println("SMCKIT_HAP\t" + row + "\t" + values(haplotype.getRawXAlleleConfig()));
                row++;
            }
        } catch (Throwable error) {
            System.out.println("SMCKIT_STATUS\tERROR");
            System.out.println("SMCKIT_ERROR_CLASS\t" + error.getClass().getName());
            System.out.println("SMCKIT_ERROR_MESSAGE\t" + String.valueOf(error.getMessage()));
        }
    }
}
