# LEET CODE STUDY NOTE

## 2022/7/5
## 729. 我的日程安排表 I
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订 ，则可以存储这个新的日程安排。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订 。

日程可以用一对整数 start 和 end 表示，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end 。

实现 MyCalendar 类：

MyCalendar() 初始化日历对象。
boolean book(int start, int end) 如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true 。否则，返回 false 并且不要将该日程安排添加到日历中。

`示例：`
```
输入：
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
输出：
[null, true, false, true]

解释：
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False ，这个日程安排不能添加到日历中，因为时间 15 已经被另一个日程安排预订了。
myCalendar.book(20, 30); // return True ，这个日程安排可以添加到日历中，因为第一个日程安排预订的每个时间都小于 20 ，且不包含时间 20 。

```
`分析`
***可以申明一个Dictionary<int,int> 来记录日历区间，key表示日程开始，vaule表示日程结束。每次book遍历calendar 判断start和end是否在已有区间则返回false。***

`c#实现`
```
public class MyCalendar {
    private Dictionary<int,int>calendar = new Dictionary<int,int>();
    public MyCalendar() {
        calendar.Clear();
    }
    
    public bool Book(int start, int end) {
        if(calendar.Count == 0){
            calendar.Add(start, end);
            return true;
        }
        
        //遍历遍历calendar
        //如果start比value还大或者end比key还小,则没有重合
        //否则重合
        // int minKey = (from d in calendar orderby d.Key ascending select d.Key).First();
        // int maxValue = (from d in calendar orderby d.Value ascending select d.Key).Last();
        foreach( KeyValuePair<int, int> kvp in calendar ){
            if(start > kvp.Key && start < kvp.Value ){
                return false;
            }else if(end > kvp.Key && end < kvp.Value){
                return false;
            }else if(start <= kvp.Key && end >= kvp.Value){
                return false;
            }
        }
        calendar.Add(start, end);
        return true;
    }
}
```


***

## 2022/7/4
## 1200. 最小绝对差
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
***
`分析`
***题目要求输出值升序排列，为避免找到结果后再排序，可以先对arr进行一次sort。定义最小差valmin,返回数组res，然后遍历一次arr 将当前下表i的值与i+1比较，如果差值比记录的小，则清空res并添加i和i+1 如果差值和记录一样，则添加i和i+1。最后返回res就是要求数组。***

`c#实现`
```
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
```
