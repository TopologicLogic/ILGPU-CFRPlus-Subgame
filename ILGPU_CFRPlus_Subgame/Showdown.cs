
using ILGPU.Runtime;
using ILGPU;
using System;

namespace ILGPU_CFRPlus_Subgame
{

    public class Showdown : Node
    {
        //public readonly double _utility;
        //public readonly  int _player;
        public readonly int _size;

        private MemoryBuffer1D<double, Stride1D.Dense> _ev;
        private MemoryBuffer1D<double, Stride1D.Dense> _utility;

        
        Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<uint>> SumMult;
        private static void Kernel_SumMult(Index1D z, ArrayView<double> ev, ArrayView<double> op, ArrayView<double> utility, ArrayView<uint> rank)
        {
            for (int j = 0; j < op.Length; j++)
            {
                if (j != z)
                {
                    if (rank[z] > rank[j])
                    {
                        ev[z] += utility[0] * op[j];
                    }
                    else if (rank[z] < rank[j])
                    {
                        ev[z] -= utility[0] * op[j];
                    }
                }
            }
            ev[z] /= (op.Length - 1);
        }

        public override MemoryBuffer1D<double, Stride1D.Dense> Train(Accelerator accelerator, int player, TrainData td, MemoryBuffer1D<double, Stride1D.Dense> op)
        {

            //Kernel_SumMult(Index1D z, ArrayView<double> ev, ArrayView<double> op, ArrayView<double> utility, ArrayView<double> rank)

            _ev.MemSetToZero();
            
            SumMult(_size, _ev.View, op.View, _utility.View, td.rank.View);

            accelerator.Synchronize();

            return _ev;
        }

        public override double[] BestResponse(int player, TrainData td, double[] op)
        {
            double[] ev = new double[op.Length];

            uint[] rank = td.rank.GetAsArray1D();
            double utility = _utility.GetAsArray1D()[0];

            for (int i = 0; i < op.Length; i++)
            {
                for (int j = 0; j < op.Length; j++)
                {
                    if (j != i)
                    {
                        if (rank[i] > rank[j])
                        {
                            ev[i] += utility * op[j];
                        }
                        else if (rank[i] < rank[j])
                        {
                            ev[i] -= utility * op[j];
                        }
                    }
                }
                ev[i] /= (op.Length - 1);

            }

            return ev;
        }

        public Showdown(Accelerator accelerator, int size,  double utility) : base(NodeType.Showdown)
        {
            //_utility = utility;
            //_player = player;
            _size = size;

            Console.WriteLine("Showdown: " + utility);

            _ev = accelerator.Allocate1D<double>(size);
            _ev.MemSetToZero();

            _utility = accelerator.Allocate1D<double>(new double[] {utility});

            SumMult = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<uint>>(Kernel_SumMult);
        }

    }
}
