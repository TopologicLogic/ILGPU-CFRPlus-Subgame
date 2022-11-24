using HoldemHand;
using System;
using ILGPU;
using ILGPU.Runtime;
using System.Collections.Generic;


namespace ILGPU_CFRPlus_Subgame
{
    internal class Program
    {

        // Bet actions by bet round as % of pot. Anything over the player stack (i.e. infinity) being all-in.
        static List<double[]> abstraction = new List<double[]> { new double[] { 0.33, 0.5, 0.75, 1, 1.25, 2.0, double.PositiveInfinity },
                                                                 new double[] { 0.25, 0.5, 1, double.PositiveInfinity },
                                                                 new double[] { 0.25, double.PositiveInfinity },
                                                                 new double[] { 1, double.PositiveInfinity }};

        //static List<double[]> abstraction = new List<double[]> { new double[] { 1,  double.PositiveInfinity },
        //                                                         new double[] { 1,  double.PositiveInfinity },
        //                                                         new double[] { 1,  double.PositiveInfinity },
        //                                                         new double[] { 1,  double.PositiveInfinity }};

        //static List<double[]> abstraction = new List<double[]> { new double[] { 1 },
        //                                                         new double[] { 1 },
        //                                                         new double[] { 1 },
        //                                                         new double[] { 1 }};

        //static List<double[]> abstraction = new List<double[]> { new double[] { 0.5 },
        //                                                         new double[] { 0.5 },
        //                                                         new double[] { 0.5 },
        //                                                         new double[] { 0.5 }};

        public static double getBestResponseValue(int size, int turn, TrainData td, Node game)
        {
            double[] op = new double[size];
            for (int i = 0; i < op.Length; i++)
                op[i] = 1.0;
            double[] ev = game.BestResponse(turn, td, op);
            double sum = 0;
            for (int i = 0; i < ev.Length; i++)
                sum += ev[i];
            return sum / size;
        }

        public static double getExploitability(int size, TrainData td, Node game)
        {
            double br0 = getBestResponseValue(size, 0, td, game);
            double br1 = getBestResponseValue(size, 1, td, game);
            return (br0 + br1) / 2;
        }

        public static Decision buildTreeHUL(Accelerator accelerator, int size, int turn, int betround, double pot, string hand_history)
        {
            bool op_checked = false;

            if (hand_history.Length > 0 && hand_history[hand_history.Length - 1] == 'k') op_checked = true;

            Console.WriteLine(hand_history + " : " + pot);

            double bv = 50;

            switch (betround)
            {
                case 0:
                    // Check or Bet.
                    if (op_checked)
                        return new Decision(hand_history, accelerator, size, turn, new Showdown(accelerator, size, pot / 2), buildTreeHUL(accelerator, size, turn ^ 1, betround + 1, pot + bv, hand_history + "b"));
                    return new Decision(hand_history, accelerator, size, turn, buildTreeHUL(accelerator, size, turn ^ 1, betround, pot, hand_history + "k"), buildTreeHUL(accelerator, size, turn ^ 1, betround + 1, pot + bv, hand_history + "b"));
                case 3:
                    // Fold or Call
                    return new Decision(hand_history, accelerator, size, turn, new Fold(accelerator, size, turn, (pot - bv) / 2), new Showdown(accelerator, size, (pot + bv) / 2));
                default:
                    // Fold, Call, or Raise
                    return new Decision(hand_history, accelerator, size, turn, new Fold(accelerator, size, turn, (pot - bv) / 2), new Showdown(accelerator, size, (pot + bv) / 2), buildTreeHUL(accelerator, size, turn ^ 1, betround + 1, pot + bv * 2, hand_history + "r"));
            }
        }

        public static Decision buildTree(Accelerator accelerator, int size, double[] stacks, int turn, int betround, double pot, double last_bet, string hand_history)
        {
            //switch (hand_history)
            //{
            //    case "b":
            //    case "br":
            //    case "bra":
            //        Console.WriteLine(hand_history + " | " + stacks[0] + " - " + stacks[1] + " | " + last_bet + " -> " + pot);
            //        break;
            //}

            Console.WriteLine(hand_history + " | " + stacks[0] + " - " + stacks[1] + " | " + last_bet + " -> " + pot);

            List<Node> actions = new List<Node> { };

            double starting_stacks = (stacks[turn] + stacks[turn ^ 1] + pot) / 2;

            if (last_bet >= stacks[turn] || stacks[turn ^ 1] <= 0.0) // Op went all-in
            {
                // Fold or call.
                actions.Add(new Fold(accelerator, size, turn, starting_stacks - stacks[turn]));
                actions.Add(new Showdown(accelerator, size, (pot + stacks[turn]) / 2));
                return new Decision(hand_history, accelerator, size, turn, actions.ToArray());
            }

            switch (betround)
            {
                case 0:
                    // Check or Bet.

                    if (hand_history.Length > 0 && hand_history[hand_history.Length - 1] == 'k') // Op checked
                        actions.Add(new Showdown(accelerator, size, pot / 2)); 
                    else
                        actions.Add(buildTree(accelerator, size, stacks, turn ^ 1, betround, pot, 0, hand_history + "k"));

                    for (int i = 0; i < abstraction[betround].Length; i++)
                    {
                        double bv = Math.Floor(pot * abstraction[betround][i]);

                        if (bv >= stacks[turn]) // All-in
                        {
                            bv = stacks[turn];
                            double[] newstacks = (double[])stacks.Clone(); newstacks[turn] = 0.0;
                            actions.Add(buildTree(accelerator, size, newstacks, turn ^ 1, betround + 1, pot + bv, bv, hand_history + "a"));
                            break;
                        }
                        else
                        {
                            double[] newstacks = (double[])stacks.Clone(); newstacks[turn] -= bv;
                            actions.Add(buildTree(accelerator, size, newstacks, turn ^ 1, betround + 1, pot + bv, bv, hand_history + "b"));
                        }
                        
                    }
                    break;

                case 3:
                    // Fold or Call

                    actions.Add(new Fold(accelerator, size, turn, starting_stacks - stacks[turn]));
                    actions.Add(new Showdown(accelerator, size, (pot + last_bet) / 2));
                    break;

                default:
                    // Fold, Call, or Raise
                    
                    actions.Add(new Fold(accelerator, size, turn, starting_stacks - stacks[turn]));
                    actions.Add(new Showdown(accelerator, size, (pot + last_bet) / 2));

                    for (int i = 0; i < abstraction[betround].Length; i++)
                    {
                        double bv = Math.Floor(pot * abstraction[betround][i]);

                        double[] newstacks = (double[])stacks.Clone();
                        //newstacks[turn] -= last_bet;

                        if (bv >= stacks[turn]) // All-in
                        {
                            bv = stacks[turn];
                            newstacks[turn] = 0.0;
                            actions.Add(buildTree(accelerator, size, newstacks, turn ^ 1, betround + 1, pot + bv, bv, hand_history + "a"));
                            break;
                        }
                        else
                        {
                            newstacks[turn] -= bv;
                            actions.Add(buildTree(accelerator, size, newstacks, turn ^ 1, betround + 1, pot + bv, bv, hand_history + "r"));
                        }
                    }
                    break;

            }

            return new Decision(hand_history, accelerator, size, turn, actions.ToArray());

        }

        static void Main(string[] args)
        {
            // Setup ILGPU
            Context context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());

            foreach (Device device in context.Devices)
                Console.WriteLine(device);

            Accelerator accelerator = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);


            // Setup the training data.
            TrainData td = new TrainData();

            // Deal yourself a board.
            ulong board = HoldemHand.Hand.ParseHand("Ac 2d 4s Kh 9d"); // HoldemHand.Hand.RandomHand(0UL, 5);

            ulong[] pockets = new ulong[1081];
            uint[] rank = new uint[1081];
            double[] op = new double[1081];

            ulong count = 0;
            // Every two card hand
            foreach (ulong pocket in Hand.Hands(0UL, board, 2))
            {
                pockets[count] = pocket;
                rank[count] = HoldemHand.Hand.Evaluate(pocket | board);
                op[count] = 1;
                count++;
            }

            Console.WriteLine(count);

            td.rank = accelerator.Allocate1D<uint>(rank);
            td.weight = accelerator.Allocate1D<double>(1);
            MemoryBuffer1D<double, Stride1D.Dense> mop = accelerator.Allocate1D<double>(op);


            // Build the game tree.

            //Decision game = buildTreeHUL(accelerator, op.Length, 1, 0, 150, "");

            Decision game = buildTree(accelerator, op.Length, new double[] { 1000 - 50, 1000 - 50 }, 1, 0, 100, 0, "");



            // Do some CFR+

            const int delay = 0;

            DateTime dt = DateTime.Now;

            for (int i = 0; i < 2000; i++)
            {
                // The update weight that's used with CFR+ is set here.
                td.weight.CopyFromCPU(new double[] { Math.Pow(Math.Max(0, i - delay + 1), 2) });       


                // Periodically show the resulting strategy.
                if (i % 10 == 0)
                {
                    Console.WriteLine(i + ": " + (DateTime.Now.Subtract(dt).TotalSeconds / i));
                    //Console.WriteLine(getExploitability(op.Length, td, game).ToString());

                    string prefix = i + ": " + getExploitability(op.Length, td, game).ToString();

                    //Console.WriteLine(prefix);

                    string output = "";

                    Decision tempd = (Decision)game; //._children[1];

                    double[,] strat = tempd.getNormalizedStrategies();

                    for (int j = 0; j < op.Length; j++)
                    {
                        output += prefix + " " +
                              HoldemHand.Hand.MaskToString(pockets[j]) + " - " +
                              HoldemHand.Hand.MaskToString(board) + ": ";

                        for (int q = 0; q < strat.GetLength(1); q++)
                            output += Math.Round(strat[j, q], 4) + " ";

                        output += "\n";
                    }

                    Console.WriteLine(output);

                }

                game.Train(accelerator, 0, td, mop);
                game.Train(accelerator, 1, td, mop);

            }

            //// Show the resulting strategy and exploitability.
            //// This doesn't use ILGPU so it's slow.
            //string prefix = getExploitability(op.Length, td, game).ToString();

            ////Console.WriteLine(prefix);

            //string output = "";

            //double[,] strat = game.getNormalizedStrategies();

            //for (int j = 0; j < op.Length; j++)
            //{
            //    output += prefix + " " +
            //          HoldemHand.Hand.MaskToString(pockets[j]) + " - " +
            //          HoldemHand.Hand.MaskToString(board) + ": ";

            //    for (int q = 0; q < strat.GetLength(1); q++)
            //        output += Math.Round(strat[j, q], 4) + " ";

            //    output += "\n";
            //}
            
            //Console.WriteLine(output);


            //accelerator.Dispose();
            //context.Dispose();



        }
    }
}
