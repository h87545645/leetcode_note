LEET CODE STUDY NOTE

2022/7/41200. 最小绝对差
给你个整数数组 arr，其中每个元素都 不相同。
请你找到所有具有最小绝对差的元素对，并且按升序的顺序返回。
示例 1：
输入：arr = [4,2,1,3]
输出：[[1,2],[2,3],[3,4]]
示例 2：
输入：arr = [1,3,6,10,15]
输出：[[1,3]]
示例 3：
输入：arr = [3,8,-10,23,19,-4,-14,27]
输出：[[-14,-10],[19,23],[23,27]]
提示：
2 <= arr.length <= 10^5
-10^6 <= arr[i] <= 10^6
题目要求输出值升序排列，为避免找到结果后再排序，可以先对arr进行一次sort。定义最小差valmin,返回数组res，然后遍历一次arr 将当前下表i的值与i+1比较，如果差值比记录的小，则清空res并添加i和i+1 如果差值和记录一样，则添加i和i+1。最后返回res就是要求数组。
c#实现

public class Solution {
    public IList<IList<int>> MinimumAbsDifference(int[] arr) {
        IList<IList<int>> res = new List<IList<int>>();
        int valMin = -1;
        int n = arr.Length;
        Array.Sort(arr);
        for(int i = 0; i < n; i++){
            if(i == n - 1)break;
                int val = Math.Abs(arr[i] - arr[i+1]);
                if(valMin < 0 || val < valMin){
                    valMin = val;
                    res.Clear();
                    List<int> temp = new  List<int>();
                    temp.Add(arr[i]);
                    temp.Add(arr[i + 1]);
 
                    res.Add(temp);
                }
                else if(val == valMin){
                    
                    List<int> temp = new  List<int>();
                    temp.Add(arr[i]);
                    temp.Add(arr[i + 1]);
  
                    res.Add(temp);
                }

        }
        return res;
    }
}