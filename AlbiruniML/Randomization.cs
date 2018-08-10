using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{

    public class Mash
    {
        public static uint n = 0xefc8249d;
        public static double mash(string data)
        {
            double h;
            for (var i = 0; i < data.Length; i++)
            {
                n += ((uint)data[i]);
                h = 0.02519603282416938 * n;
                n = ((uint)h >> 2);
                h -= n;
                h *= n;
                n = ((uint)h >> 2);
                h -= n;
                n += (uint)h * 0x10000000;

            }
            return ((double)((uint)n >> 0)) * 2.3283064365386963e-10; // 2^-32
        }
    }
    public class AleaRandomization
    {
        double s0 = 0;
        double s1 = 0;
        double s2 = 0;
        double c = 1;
        public AleaRandomization(string seed)
        {
            s0 = Mash.mash(" ");
            s1 = Mash.mash(" ");
            s2 = Mash.mash(" ");
            s0 -= Mash.mash(seed);
            if (s0 < 0)
            {
                s0 += 1;
            }
            s1 -= Mash.mash(seed);
            if (s1 < 0)
            {
                s1 += 1;
            }
            s2 -= Mash.mash(seed);
            if (s2 < 0)
            {
                s2 += 1;
            }

            var date = DateTime.Now.ToString();
            s0 -= Mash.mash(date);
            if (s0 < 0)
            {
                s0 += 1;
            }
            s1 -= Mash.mash(date);
            if (s1 < 0)
            {
                s1 += 1;
            }
            s2 -= Mash.mash(date);
            if (s2 < 0)
            {
                s2 += 1;
            }
            Mash.n = 0xefc8249d;
        }
        public double random()
        {
            var t = 2091639.0 * s0 + c * 2.3283064365386963e-10; // 2^-32
            s0 = s1;
            s1 = s2;
            return s2 = t - (c = (int)t | 0);
        }
        public uint uint32()
        {
            return (uint)(random() * 0x1000000000);
        }
        public double fract53()
        {
            return random() +
         ((int)(random() * 0x200000) | 0) * 1.1102230246251565e-16;
        }
    }
   
}
