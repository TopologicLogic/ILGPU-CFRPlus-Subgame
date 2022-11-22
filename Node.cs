
using ILGPU;
using ILGPU.Runtime;

namespace ILGPU_CFRPlus_Subgame
{
    public enum NodeType
    {
        Fold,
        Decision,
        Showdown
    }

    public struct TrainData
    {
        public MemoryBuffer1D<uint, Stride1D.Dense> rank;
        public MemoryBuffer1D<int, Stride1D.Dense> train_player;
        public MemoryBuffer1D<double, Stride1D.Dense> weight;
        //public MemoryBuffer1D<int, Stride1D.Dense> iterationCount;
        //public int iterationCount;
    }

    public abstract class Node
    {
        public NodeType type { get; }

        public abstract MemoryBuffer1D<double, Stride1D.Dense> Train(Accelerator accelerator, int player, TrainData td, MemoryBuffer1D<double, Stride1D.Dense> op);

        public abstract double[] BestResponse(int player, TrainData td, double[] op);

        public Node(NodeType node_type)
        {
            type = node_type;
        }
    }
}
