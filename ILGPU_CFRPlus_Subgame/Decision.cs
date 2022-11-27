using ILGPU.Runtime;
using ILGPU;
using System;
using System.Threading.Tasks;
using System.Linq;

namespace ILGPU_CFRPlus_Subgame
{
    public class Decision : Node
    {
        public readonly Node[] _children;
        private readonly int _player;
        private readonly int _size;
        private readonly string _hh; // Just used to display/debugging.

        private MemoryBuffer2D<double, Stride2D.DenseX> _cfr;
        private MemoryBuffer2D<double, Stride2D.DenseX> _strategy;
        private MemoryBuffer1D<double, Stride1D.Dense> _newop;

        private MemoryBuffer1D<double, Stride1D.Dense> _ev;
        private MemoryBuffer2D<double, Stride2D.DenseX> _u;
        private MemoryBuffer2D<double, Stride2D.DenseX> _s;
        private MemoryBuffer1D<int, Stride1D.Dense> _idx;

        private Action<Index1D, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>> SumMult;
        private static void Kernel_SumMult(Index1D z, ArrayView<double> evs, ArrayView2D<double, Stride2D.DenseX> s, ArrayView2D<double, Stride2D.DenseX> u)
        {
            //for (int a = 0; a < u.Stride.YStride; a++)
            //    evs[z] += s[z, a] * u[a, z];

            //int csize = ((Index2D)s.Extent).Y;

            int csize = (int)(s.Length / s.Stride.YStride);

            for (int a = 0; a < csize; a++)
                evs[z] += s[z, a] * u[a, z];
        }

        private Action<Index1D, ArrayView<int>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>> TransposeCopy;
        private static void Kernel_TransposeCopy(Index1D z, ArrayView<int> indx, ArrayView<double> ev, ArrayView2D<double, Stride2D.DenseX> u)
        {
            u[indx[0], z] = ev[z];
        }

        private Action<Index1D, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>> PositiveSum;
        private static void Kernel_PositiveSum(Index1D z, ArrayView<double> sums, ArrayView2D<double, Stride2D.DenseX> cfr)
        {
            //int csize = ((Index2D)cfr.Extent).Y;
            int csize = (int)(cfr.Length / cfr.Stride.YStride);
            for (int a = 0; a < csize; a++)
                if (cfr[z, a] > 0) sums[z] += cfr[z, a];
        }

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>> CalcCFR;
        private static void Kernel_CalcCFR(Index1D z, ArrayView2D<double, Stride2D.DenseX> cfr, ArrayView<double> ev, ArrayView2D<double, Stride2D.DenseX> u)
        {
            //for (int a = 0; a < _children.Length; a++)
            //{
            //    _cfr[i][a] += u[a][i] - ev[i];
            //    _cfr[i][a] = Math.Max(0, _cfr[i][a]);
            //}

            //int csize = ((Index2D)cfr.Extent).Y;
            int csize = (int)(cfr.Length / cfr.Stride.YStride);

            for (int a = 0; a < csize; a++)
            {
                cfr[z, a] += u[a, z] - ev[z];
                if (cfr[z, a] < 0) cfr[z, a] = 0;
            }
        }

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>> GetCurrentStrat;
        private static void Kernel_GetCurrentStrat(Index1D z, ArrayView2D<double, Stride2D.DenseX> s, ArrayView2D<double, Stride2D.DenseX> cfr)
        {
            //double[] psum = new double[size];
            //double[,] s = new double[size, _children.Length];

            //for (int hand = 0; hand < size; hand++)
            //{
            //    for (int a = 0; a < _children.Length; a++)
            //    {
            //        if (_cfr[hand][a] > 0)
            //            psum[hand] += _cfr[hand][a];
            //    }
            //    if (psum[hand] > 0)
            //    {
            //        for (int a = 0; a < _children.Length; a++)
            //            s[hand, a] = _cfr[hand][a] > 0 ? _cfr[hand][a] / psum : 0;
            //    }
            //    else
            //    {
            //        for (int a = 0; a < _children.Length; a++)
            //            s[hand, a] = 1.0 / _children.Length;
            //    }
            //}
            //return s;

            //int csize = ((Index2D)cfr.Extent).Y;
            int csize = (int)(cfr.Length / cfr.Stride.YStride);

            double sum = 0;
            for (int a = 0; a < csize; a++)
                if (cfr[z, a] > 0) sum += cfr[z, a];

            if (sum > 0)
            {
                for (int a = 0; a < csize; a++)
                    if (cfr[z, a] > 0) s[z, a] = cfr[z, a] / sum; else s[z, a] = 0;
            }
            else
            {
                for (int a = 0; a < csize; a++)
                    s[z, a] = 1.0 / csize;
            }
        }

        private Action<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>> CalcStrat;
        private static void Kernel_CalcStrat(Index1D z, ArrayView2D<double, Stride2D.DenseX> strat, ArrayView<double> op, ArrayView2D<double, Stride2D.DenseX> s, ArrayView<double> weight)
        {

            //for (int i = 0; i < op.Length; i++)
            //{
            //    for (int a = 0; a < _children.Length; a++)
            //        _strategy[i][a] += op[i] * s[i][a] * weight;
            //}
            //int csize = ((Index2D)strat.Extent).Y;
            int csize = (int)(strat.Length / strat.Stride.YStride);

            for (int a = 0; a < csize; a++)
                strat[z, a] += op[z] * s[z, a] * weight[0];


        }

        private Action<Index1D, ArrayView<int>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>> CalcNewOp;
        private static void Kernel_CalcNewOp(Index1D z, ArrayView<int> idx, ArrayView<double> newop, ArrayView2D<double, Stride2D.DenseX> s, ArrayView<double> op)
        {
            newop[z] = s[z, idx[0]] * op[z];
        }

        private Action<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>> CopyOrAdd;
        private static void Kernel_CopyOrAdd(Index1D z, ArrayView<int> idx, ArrayView<double> a, ArrayView<double> b)
        {
            if (idx[0] == 0) a[z] = b[z]; else a[z] += b[z];
        }


        public override MemoryBuffer1D<double, Stride1D.Dense> Train(Accelerator accelerator, int player, TrainData td, MemoryBuffer1D<double, Stride1D.Dense> op)
        {

            //double[,] s = getCurrentStrategies(_size);
            //for (int i = 0; i < op.Length; i++)
            //{
            //    double[] t = getCurrentStrategy(i);
            //    for (int b = 0; b < t.Length; b++)
            //        s[i, b] = t[b];
            //}

            _s.MemSetToZero();

            GetCurrentStrat(_size, _s.View, _cfr.View);

            accelerator.Synchronize();

            _ev.MemSetToZero();
            //double[] ev = new double[op.Length];

            if (_player == player)
            {
                // double[,] u = new double[_children.Length, op.Length];

                //for (int a = 0; a < _children.Length; a++)
                //{
                //    u[a] = _children[a].Train(player, td, op);
                //    for (int i = 0; i < op.Length; i++)
                //        ev[i] += s[i][a] * u[a][i];
                //}

                //for (int a = 0; a < _children.Length; a++)
                //{
                //    double[] t = _children[a].Train(accelerator, player, td, op);
                //    for (int b = 0; b < t.Length; b++)    
                //        u[a, b] = t[b];
                //}

                //MemoryBuffer1D<double, Stride1D.Dense> t;

                _u.MemSetToZero();

                for (int a = 0; a < _children.Length; a++)
                {
                    //t = _children[a].Train(accelerator, player, td, op);

                    // Transpose evs => u[a,evs]
                    _idx.CopyFromCPU(new int[] { a });
                    TransposeCopy(_size, _idx.View, _children[a].Train(accelerator, player, td, op).View, _u.View);
                }


                accelerator.Synchronize();

                SumMult(_size, _ev.View, _s.View, _u.View); 
                
                accelerator.Synchronize();

                CalcCFR(_size, _cfr.View, _ev.View, _u.View);

                accelerator.Synchronize();

            }
            else
            {
                //const double delay = 0;
                //double weight = Math.Pow(Math.Max(0, td.iterationCount - delay + 1), 2);

                //for (int i = 0; i < op.Length; i++)
                //{
                //    for (int a = 0; a < _children.Length; a++)
                //        _strategy[i][a] += op[i] * s[i][a] * weight;
                //}

                CalcStrat(_size, _strategy.View, op.View, _s.View, td.weight.View);

                accelerator.Synchronize();


                for (int a = 0; a < _children.Length; a++)
                {
                    //double[] newop = new double[op.Length];
                    //for (int i = 0; i < op.Length; i++)
                    //    newop[i] = s[i][a] * op[i];

                    _idx.CopyFromCPU(new int[] { a });

                    CalcNewOp(_size, _idx.View, _newop.View, _s.View, op.View);

                    accelerator.Synchronize();

                    CopyOrAdd(_size, _idx.View, _ev.View, _children[a].Train(accelerator, player, td, _newop).View);

                    accelerator.Synchronize();
                }



            }

            return _ev;

        }

        public override double[] BestResponse(int player, TrainData td, double[] op)
        {
            double[] ev = new double[op.Length];

            if (_player == player)
            {
                ev = _children[0].BestResponse(player, td, op);

                for (int i = 1; i < _children.Length; i++)
                    max(ev, _children[i].BestResponse(player, td, op));

            }
            else
            {
                double[] newop = new double[op.Length];

                //for (int a = 0; a < _children.Length; a++)
                //{
                //    for (int i = 0; i < op.Length; i++)
                //    {
                //        double[] s = getNormalizedStrategy(i);
                //        newop[i] = s[a] * op[i];
                //    }

                //    double[] br = _children[a].BestResponse(player, td, newop);

                //    if (a == 0) ev = br; else add(ev, br);
                //}

                double[,] s = getNormalizedStrategies();

                for (int a = 0; a < _children.Length; a++)
                {
                    Parallel.For(0, op.Length, i =>
                    {
                        newop[i] = s[i, a] * op[i];
                    });
                    //for (int i = 0; i < op.Length; i++)
                    //    newop[i] = s[i, a] * op[i];

                    double[] br = _children[a].BestResponse(player, td, newop);

                    if (a == 0) ev = br; else add(ev, br);
                }
            }

            return ev;
        }

        private static void max(double[] a, double[] b)
        {
            Parallel.For(0, a.Length, i =>
            {
                if (a[i] < b[i]) a[i] = b[i];
            });
            
            //for (int i = 0; i < a.Length; i++)
            //    if (a[i] < b[i]) a[i] = b[i];
        }

        private static void add(double[] a, double[] b)
        {
            Parallel.For(0, a.Length, i =>
            {
                a[i] += b[i];
            });

            //for (int i = 0; i < a.Length; i++)
            //    a[i] += b[i];
        }

        //public void copyOrAdd(bool copy, double[] a, double [] b)
        //{
        //    if (copy)
        //    {
        //        for (int i = 0; i < a.Length; i++)
        //            a[i] = b[i];
        //    }
        //    else
        //    {
        //        for (int i = 0; i < a.Length; i++)
        //            a[i] += b[i];
        //    }
        //}

        //public double[,] getCurrentStrategies(int size)
        //{
        //    double[] psum = new double[size];
        //    double[,] s = new double[size, _children.Length];

        //    for (int hand = 0; hand < size; hand++)
        //    { 
        //        for (int a = 0; a < _children.Length; a++)
        //        {
        //            if (_cfr[hand][a] > 0)
        //                psum[hand] += _cfr[hand][a];
        //        }
        //        if (psum[hand] > 0)
        //        {
        //            for (int a = 0; a < _children.Length; a++)
        //                s[hand, a] = _cfr[hand][a] > 0 ? _cfr[hand][a] / psum : 0;
        //        }
        //        else
        //        {
        //            for (int a = 0; a < _children.Length; a++)
        //                s[hand, a] = 1.0 / _children.Length;
        //        }
        //    }
        //    return s;
        //}

        //private double[] getNormalizedStrategy(int hand)
        //{
        //    double sum = 0;

        //    double[,] strat = _strategy.GetAsArray2D();

        //    for (int a = 0; a < _children.Length; a++)
        //        sum += strat[hand, a];

        //    double[] s = new double[_children.Length];

        //    if (sum > 0)
        //    {
        //        for (int a = 0; a < _children.Length; a++)
        //            s[a] = strat[hand, a] / sum;
        //    }
        //    else
        //    {
        //        for (int a = 0; a < _children.Length; a++)
        //            s[a] = 1.0 / _children.Length;
        //    }

        //    return s;
        //}

        public double[,] getNormalizedStrategies()
        {

            double[,] strat = _strategy.GetAsArray2D();

            Parallel.For(0, strat.GetLength(0), hand =>
            {

                //for (int hand = 0; hand < strat.GetLength(0); hand++)
                //{

                double sum = 0.0;

                for (int a = 0; a < _children.Length; a++)
                    sum += strat[hand, a];

                if (sum > 0)
                {
                    for (int a = 0; a < _children.Length; a++)
                        strat[hand, a] /= sum;
                }
                else
                {
                    for (int a = 0; a < _children.Length; a++)
                        strat[hand, a] = 1.0 / _children.Length;
                }

            });

            return strat;
        }



        public Decision(string hh, Accelerator accelerator, int size, int player, params Node[] children) : base(NodeType.Decision)
        {
            _player = player;
            _size = size;
            _cfr = accelerator.Allocate2DDenseX<double>(new Index2D(size, children.Length));
            _strategy = accelerator.Allocate2DDenseX<double>(new Index2D(size, children.Length));
            _children = children;
            _hh = hh;

            _ev = accelerator.Allocate1D<double>(size);
            _u = accelerator.Allocate2DDenseX<double>(new Index2D(_children.Length, size));
            _s = accelerator.Allocate2DDenseX<double>(new Index2D(size, _children.Length));
            _idx = accelerator.Allocate1D<int>(1);

            _newop = accelerator.Allocate1D<double>(size);

            _cfr.MemSetToZero();
            _strategy.MemSetToZero();
            _ev.MemSetToZero();
            _u.MemSetToZero();
            _s.MemSetToZero();
            _idx.MemSetToZero();
            _newop.MemSetToZero();


            //Console.WriteLine(((Index2D)_s.Extent).Y + " - " + ((Index2D)_strategy.Extent).Y);


            TransposeCopy = accelerator.LoadAutoGroupedStreamKernel< Index1D, ArrayView<int>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>> (Kernel_TransposeCopy);
            SumMult = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(Kernel_SumMult);
            PositiveSum = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>>(Kernel_PositiveSum);
            CalcCFR = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>>(Kernel_CalcCFR);
            GetCurrentStrat = accelerator.LoadAutoGroupedStreamKernel< Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView2D<double, Stride2D.DenseX>>(Kernel_GetCurrentStrat);

            CalcStrat = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>>(Kernel_CalcStrat);
            CalcNewOp = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<double>, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>>(Kernel_CalcNewOp);

            CopyOrAdd = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<double>, ArrayView<double>>(Kernel_CopyOrAdd);
        }

    }
}
