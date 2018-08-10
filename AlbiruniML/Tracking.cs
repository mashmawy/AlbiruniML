using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public static partial class Ops
    {


        public static MemoryInfo memory()
        {
            return ENV.engine.memory();
        }
        public static void tidy( Action fn, bool gradMode = false)
        {
            ENV.engine.tidy(  fn, gradMode);

        }
        public static List<Tensor> tidy(   Func<List<Tensor>> fn, bool gradMode = false)
        {
            return ENV.engine.tidy(  fn, gradMode);
        }
        public static Tensor tidy(  Func<Tensor> fn, bool gradMode = false)
        {
            return ENV.engine.tidy(  fn, gradMode);
        }
        public static void tidy(string name, Action fn, bool gradMode = false)
        {
            ENV.engine.tidy(name, fn, gradMode);
            
        }
        public static List<Tensor> tidy(string name, Func<List<Tensor>> fn, bool gradMode = false)
        {
           return ENV.engine.tidy(name, fn, gradMode);
        }
        public static Tensor tidy(string name, Func<Tensor> fn, bool gradMode = false)
        {
            return ENV.engine.tidy(name, fn, gradMode);
        }
        public static Tensor keep(this Tensor result)
        {
            return ENV.engine.keep(result);
        }
    }
}
