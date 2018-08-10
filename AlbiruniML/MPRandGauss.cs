
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlbiruniML
{
    public class MPRandGauss
    {
        private double mean;
        private double stdDev;
        private double nextVal;
        private Nullable<bool> truncated;
        private Nullable<double> upper;
        private Nullable<double> lower;
        private AleaRandomization random;
        public MPRandGauss(double mean, double stdDeviation, Nullable<bool> truncated = null, Nullable<double> seed = null)
        {
            this.mean = mean;
            this.stdDev = stdDeviation;
            this.nextVal = double.NaN;
            this.truncated = truncated;
            if (this.truncated == true)
            {
                this.upper = this.mean + this.stdDev * 2;
                this.lower = this.mean - this.stdDev * 2;
            }
            var seedValue = seed.HasValue ? seed : new Random(335).NextDouble();
            this.random = new AleaRandomization(seedValue.ToString());
        }


        public double nextValue()
        {
            if (!double.IsNaN(this.nextVal))
            {
                var value = this.nextVal;
                this.nextVal = double.NaN;
                return value;
            }

            double resultX=0;
            double resultY=0;
            bool isValid=false;
            while (!isValid)
            {
                double v1;
                double v2;
                double s;
                do
                {
                    v1 = 2 * this.random.random() - 1;
                    v2 = 2 * this.random.random() - 1;
                    s = v1 * v1 + v2 * v2;
                } while (s >= 1 || s == 0);

                var mul = Math.Sqrt(-2.0 * Math.Log(s) / s);
                resultX = this.mean + this.stdDev * v1 * mul;
                resultY = this.mean + this.stdDev * v2 * mul;

                if (this.truncated==false || this.isValidTruncated(resultX))
                {
                    isValid = true;
                }
            }
            if (this.truncated==false || this.isValidTruncated(resultY))
            {
                this.nextVal = resultY;
            }
            return resultX;
        }
        private bool isValidTruncated(double value)
        {
            return value <= this.upper && value >= this.lower;
        }
         
    }
}
