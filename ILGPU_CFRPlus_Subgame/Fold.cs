using ILGPU.Runtime;
using ILGPU;
using System;
using System.Threading.Tasks;

namespace ILGPU_CFRPlus_Subgame
{
    public class Fold : Node
    {
        public readonly int _size;
        public readonly int _player;

        private MemoryBuffer1D<double, Stride1D.Dense> _ev;
        private MemoryBuffer1D<double, Stride1D.Dense> _pos_utility;
        private MemoryBuffer1D<double, Stride1D.Dense> _neg_utility;
        //private static MemoryBuffer1D<int, Stride1D.Dense> _player;



        private Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> SumMult;
        private static void Kernel_SumMult(Index1D z, ArrayView<double> ev, ArrayView<double> op, ArrayView<double> utility)
        {
            ev[z] = op[z] * utility[0];
        }

        public override MemoryBuffer1D<double, Stride1D.Dense> Train(Accelerator accelerator, int player, TrainData td, MemoryBuffer1D<double, Stride1D.Dense> op)
        {
            _ev.MemSetToZero();

            if (_player == player)
            {
                // Positive ev
                SumMult(_size, _ev.View, op.View, _neg_utility.View);
            }
            else
            {
                // Negative ev
                SumMult(_size, _ev.View, op.View, _pos_utility.View);
            }

            accelerator.Synchronize();

            return _ev;

            //double[] ev = new double[op.Length];

            //if (_player == player)
            //{
            //    for (int i = 0; i < op.Length; i++)
            //        ev[i] = op[i] * _utility;
            //}
            //else
            //{
            //    for (int i = 0; i < op.Length; i++)
            //        ev[i] = op[i] * -_utility;
            //}

            //return null;
        }

        public override double[] BestResponse(int player, TrainData td, double[] op)
        {
            double[] ev = new double[op.Length];

            //if (_player == player)
            //{
            //    for (int i = 0; i < op.Length; i++)
            //        ev[i] = op[i] * _pos_utility.GetAsArray1D()[0];
            //}
            //else
            //{
            //    for (int i = 0; i < op.Length; i++)
            //        ev[i] = op[i] * _neg_utility.GetAsArray1D()[0];
            //}

            double utility = _pos_utility.GetAsArray1D()[0];

            if (_player == player)
            {
                Parallel.For(0, op.Length, i =>
                {
                    ev[i] = op[i] * -utility;
                });
                //for (int i = 0; i < op.Length; i++)
                //ev[i] = op[i] * -utility;
            }
            else
            {
                Parallel.For(0, op.Length, i =>
                {
                    ev[i] = op[i] * utility;
                });
                //for (int i = 0; i < op.Length; i++)
                //    ev[i] = op[i] * utility;
            }


            return ev;
        }

        public Fold(Accelerator accelerator, int size, int player,  double utility) : base(NodeType.Fold)
        {
            _player = player;
            _size = size;

            Console.WriteLine("Fold: " + utility);

            _ev = accelerator.Allocate1D<double>(size);
            _ev.MemSetToZero();

            _pos_utility = accelerator.Allocate1D<double>(new double[] { utility });
            _neg_utility = accelerator.Allocate1D<double>(new double[] { -utility });

            SumMult = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(Kernel_SumMult);
        }

    }
}
