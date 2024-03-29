# LEET CODE STUDY NOTE

## 2023/12/27

## 面试题 01.01. 判定字符是否唯一

[面试题 01.01. 判定字符是否唯一](https://leetcode.cn/problems/is-unique-lcci/description/?envType=featured-list&envId=xb9lfcwi?envType=featured-list&envId=xb9lfcwi)
```
实现一个算法，确定一个字符串 s 的所有字符是否全都不同。

示例 1：

输入: s = "leetcode"
输出: false 
示例 2：

输入: s = "abc"
输出: true
限制：

0 <= len(s) <= 100 
s[i]仅包含小写字母
如果你不使用额外的数据结构，会很加分。
```

`思路`
排序

`c# 实现`
```
public class Solution {
    public bool IsUnique(string astr) {
        char[] chars = astr.ToCharArray();
        if(chars.Length == 0) return true;
        Array.Sort(chars);
        char previous = chars[0];
        for (int i = 1; i < chars.Length; i++)
        {
            if (previous == chars[i])
            {
                return false;
            }
            previous = chars[i];
        }
        return true;
    }
}
```

***

## 2023/12/14

## 454. 四数相加 II

[454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：

0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
 

示例 1：

输入：nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
输出：2
解释：
两个元组如下：
1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
示例 2：

输入：nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
输出：1
```

`思路`
哈希表记录

`c# 实现`
```
public class Solution {
    public int FourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Dictionary<int , int> dict = new  Dictionary<int , int>();
        foreach (int a in nums1)
        {
            foreach (int b in nums2)
            {
                dict.TryAdd(a+b , 0);
                dict[a+b] ++;
            }
        }
        int ans = 0;
        foreach (int a in nums3)
        {
            foreach (int b in nums4)
            {
                if (dict.ContainsKey(-a-b))
                {
                    ans += dict[-a-b];
                }
            }
        }
        return ans;
    }
}
```

***

## 2023/12/13

## 412. Fizz Buzz

[412. Fizz Buzz](https://leetcode.cn/problems/fizz-buzz/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个整数 n ，找出从 1 到 n 各个整数的 Fizz Buzz 表示，并用字符串数组 answer（下标从 1 开始）返回结果，其中：

answer[i] == "FizzBuzz" 如果 i 同时是 3 和 5 的倍数。
answer[i] == "Fizz" 如果 i 是 3 的倍数。
answer[i] == "Buzz" 如果 i 是 5 的倍数。
answer[i] == i （以字符串形式）如果上述条件全不满足。
 

示例 1：

输入：n = 3
输出：["1","2","Fizz"]
示例 2：

输入：n = 5
输出：["1","2","Fizz","4","Buzz"]
示例 3：

输入：n = 15
输出：["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
```

`思路`

`c# 实现`
```
public class Solution {

    public IList<string> FizzBuzz(int n) {
        List<string> ans = new List<string>();
        for (int i = 1; i <= n; i++)
        {
            string fb = "";
            if (i % 3 == 0)
            {
                fb += "Fizz";
            }
            if (i % 5 == 0)
            {
                fb += "Buzz";
            }
            if (fb == "")
            {
                fb = i + "";
            }
            ans.Add(fb);
        }
        return ans;
    }
}
```

***

## 2023/12/12

## 395. 至少有 K 个重复字符的最长子串

[395. 至少有 K 个重复字符的最长子串](https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个字符串 s 和一个整数 k ，请你找出 s 中的最长子串， 要求该子串中的每一字符出现次数都不少于 k 。返回这一子串的长度。

如果不存在这样的子字符串，则返回 0。

 

示例 1：

输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。
示例 2：

输入：s = "ababbc", k = 2
输出：5
解释：最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
```

`思路`
递归分治

`c# 实现`
```
public class Solution {
    public int LongestSubstring(string s, int k) {
        if (s.Length < k)
        {
            return 0;
        }
        Dictionary<char,int> counter = new Dictionary<char,int>();
        for (int i = 0; i < s.Length; i++)
        {
            counter.TryAdd(s[i],0);
            counter[s[i]]++;
        }
        foreach (KeyValuePair<char, int> kvp in counter)
        {
            if (kvp.Value < k)
            {
                int res = 0;
                foreach (string subs in s.Split(kvp.Key))
                {
                    res = Math.Max(res,LongestSubstring(subs,k));
                }
                return res;       
            }
        }
        return s.Length;
    }
}
```

***

## 2023/11/30

## 387. 字符串中的第一个唯一字符

[387. 字符串中的第一个唯一字符](https://leetcode.cn/problems/first-unique-character-in-a-string/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定一个字符串 s ，找到 它的第一个不重复的字符，并返回它的索引 。如果不存在，则返回 -1 。

 

示例 1：

输入: s = "leetcode"
输出: 0
示例 2:

输入: s = "loveleetcode"
输出: 2
示例 3:

输入: s = "aabb"
输出: -1

```

`思路`
哈希表

`c# 实现`
```
public class Solution {
    public int FirstUniqChar(string s) {
        int ans = -1;
        Dictionary<char,int> dict = new Dictionary<char, int>();
        for (int i = 0; i < s.Length; i++)
        {
            if (dict.ContainsKey(s[i]))
            {
                dict[s[i]] = -1;
            }else{
                dict.Add(s[i],i);
            }
        }
        foreach( KeyValuePair<char, int> kvp in dict ){
            if (kvp.Value != -1)
            {
                ans = kvp.Value;
                break;
            }
        }
        return ans;
    }
}
```

***


## 2023/11/23

## 384. 打乱数组

[384. 打乱数组](https://leetcode.cn/problems/shuffle-an-array/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个整数数组 nums ，设计算法来打乱一个没有重复元素的数组。打乱后，数组的所有排列应该是 等可能 的。

实现 Solution class:

Solution(int[] nums) 使用整数数组 nums 初始化对象
int[] reset() 重设数组到它的初始状态并返回
int[] shuffle() 返回数组随机打乱后的结果
 

示例 1：

输入
["Solution", "shuffle", "reset", "shuffle"]
[[[1, 2, 3]], [], [], []]
输出
[null, [3, 1, 2], [1, 2, 3], [1, 3, 2]]

解释
Solution solution = new Solution([1, 2, 3]);
solution.shuffle();    // 打乱数组 [1,2,3] 并返回结果。任何 [1,2,3]的排列返回的概率应该相同。例如，返回 [3, 1, 2]
solution.reset();      // 重设数组到它的初始状态 [1, 2, 3] 。返回 [1, 2, 3]
solution.shuffle();    // 随机返回数组 [1, 2, 3] 打乱后的结果。例如，返回 [1, 3, 2]
```

`思路`


`c# 实现`
```
public class Solution {
    int[] nums;
    int[] origin;
    public Solution(int[] nums) {
        this.nums = nums;
        origin = new int[nums.Length];
        Array.Copy(nums, origin, nums.Length);
    }
    
    public int[] Reset() {
        return origin;
    }
    
    public int[] Shuffle() {
        int[] shuffled = new int[nums.Length];
        IList<int> list = new List<int>(nums);
        // for (int i = 0; i < nums.Length; ++i) {
        //     list.Add(nums[i]);
        // }
        Random random = new Random();
        for (int i = 0; i < nums.Length; ++i) {
            int j = random.Next(list.Count);
            shuffled[i] = list[j];
            list.Remove(list[j]);
        }
        Array.Copy(shuffled, nums, nums.Length);
        return nums;
    }
}
```

***

## 2023/11/22

## 380. O(1) 时间插入、删除和获取随机元素

[380. O(1) 时间插入、删除和获取随机元素](https://leetcode.cn/problems/insert-delete-getrandom-o1/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
public class RandomizedSet {
    IList<int> nums;
    Dictionary<int, int> indices;
    Random random;
    public RandomizedSet() {
        nums = new List<int>();
        indices = new  Dictionary<int, int>();
        random = new Random();
    }
    
    public bool Insert(int val) {
        if (indices.ContainsKey(val))
        {
            return false;
        }
        int index = nums.Count;
        nums.Add(val);
        indices.Add(val,index);
        return true;
    }
    
    public bool Remove(int val) {
        if (!indices.ContainsKey(val))
        {
            return false;
        }
        int index = indices[val];
        int last = nums[nums.Count - 1];
        nums[index] = last;
        indices[last] = index;
        nums.RemoveAt(nums.Count - 1);
        indices.Remove(val);
        return true;
    }
    
    public int GetRandom() {
        int randomIndex = random.Next(nums.Count);
        return nums[randomIndex];
    }
}
```

`思路`
哈希表辅助

`c# 实现`
```
public class RandomizedSet {
    IList<int> nums;
    Dictionary<int, int> indices;
    Random random;
    public RandomizedSet() {
        nums = new List<int>();
        indices = new  Dictionary<int, int>();
        random = new Random();
    }
    
    public bool Insert(int val) {
        if (indices.ContainsKey(val))
        {
            return false;
        }
        int index = nums.Count;
        nums.Add(val);
        indices.Add(val,index);
        return true;
    }
    
    public bool Remove(int val) {
        if (!indices.ContainsKey(val))
        {
            return false;
        }
        int index = indices[val];
        int last = nums[nums.Count - 1];
        nums[index] = last;
        indices[last] = index;
        nums.RemoveAt(nums.Count - 1);
        indices.Remove(val);
        return true;
    }
    
    public int GetRandom() {
        int randomIndex = random.Next(nums.Count);
        return nums[randomIndex];
    }
}
```

***

## 2023/11/21

## 378. 有序矩阵中第 K 小的元素

[378. 有序矩阵中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。

你必须找到一个内存复杂度优于 O(n2) 的解决方案。

 

示例 1：

输入：matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
输出：13
解释：矩阵中的元素为 [1,5,9,10,11,12,13,13,15]，第 8 小元素是 13
示例 2：

输入：matrix = [[-5]], k = 1
输出：-5
```

`思路`
二分查找

`c# 实现`
```
public class Solution {
    public int KthSmallest(int[][] matrix, int k) {
        int n = matrix.Length;
        int l = matrix[0][0];
        int r = matrix[n - 1][n - 1];
        while(l < r){
            int mid = l + ((r - l) >> 1);
            if (CheckContain(matrix , mid, k , n))
            {
                r = mid;
            }else{
                l = mid + 1;
            }
        }
        return l;
    }

    private bool CheckContain(int[][] matrix , int mid , int k , int n){
        int i = n - 1;
        int j = 0;
        int num = 0;
        while(i >= 0 && j < n){
            if (matrix[i][j] <= mid)
            {
                num += i + 1;
                j ++;
            }else{
                i --;
            }
        }
        return num >= k;
    }
}
```

***

## 2023/11/20

## 371. 两整数之和

[371. 两整数之和](https://leetcode.cn/problems/sum-of-two-integers/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你两个整数 a 和 b ，不使用 运算符 + 和 - ​​​​​​​，计算并返回两整数之和。

 

示例 1：

输入：a = 1, b = 2
输出：3
示例 2：

输入：a = 2, b = 3
输出：5
```

`思路`
位运算

`c# 实现`
```
public class Solution {
    public int GetSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }
}

```

***

## 2023/11/17

## 350. 两个数组的交集 II

[350. 两个数组的交集 II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。

 

示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

```

`思路`
双指针或者哈希表

`c# 实现`
```
public class Solution {
    public int[] Intersect(int[] nums1, int[] nums2) {
        if (nums1.Length > nums2.Length)
        {
            return Intersect(nums2,nums1);
        }
        List<int> ans = new List<int>();
        Dictionary<int,int> dict = new Dictionary<int, int>();
        for (int i = 0; i < nums1.Length; i++)
        {
            dict.TryAdd(nums1[i],0);
            dict[nums1[i]]++;
        }
        for (int i = 0; i < nums2.Length; i++)
        {
            if (dict.ContainsKey(nums2[i]))
            {
                ans.Add(nums2[i]);
                dict[nums2[i]] --;
                if (dict[nums2[i]] == 0)
                {
                    dict.Remove(nums2[i]);
                }
            }
        }
        return ans.ToArray();
    }
}
```

***

## 2023/11/16

## 344. 反转字符串

[344. 反转字符串](https://leetcode.cn/problems/reverse-string/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

 

示例 1：

输入：s = ["h","e","l","l","o"]
输出：["o","l","l","e","h"]
示例 2：

输入：s = ["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

```

`思路`
双指针

`c# 实现`
```
public class Solution {
    public void ReverseString(char[] s) {
        for (int i = 0 , j = s.Length - 1; i < j; i++ , j--)
        {
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;
        }
    }
}
```

***

## 2023/11/15

## 341. 扁平化嵌套列表迭代器

[341. 扁平化嵌套列表迭代器](https://leetcode.cn/problems/flatten-nested-list-iterator/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个嵌套的整数列表 nestedList 。每个元素要么是一个整数，要么是一个列表；该列表的元素也可能是整数或者是其他列表。请你实现一个迭代器将其扁平化，使之能够遍历这个列表中的所有整数。

实现扁平迭代器类 NestedIterator ：

NestedIterator(List<NestedInteger> nestedList) 用嵌套列表 nestedList 初始化迭代器。
int next() 返回嵌套列表的下一个整数。
boolean hasNext() 如果仍然存在待迭代的整数，返回 true ；否则，返回 false 。
你的代码将会用下述伪代码检测：

initialize iterator with nestedList
res = []
while iterator.hasNext()
    append iterator.next() to the end of res
return res
如果 res 与预期的扁平化列表匹配，那么你的代码将会被判为正确。

 

示例 1：

输入：nestedList = [[1,1],2,[1,1]]
输出：[1,1,2,1,1]
解释：通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。
示例 2：

输入：nestedList = [1,[4,[6]]]
输出：[1,4,6]
解释：通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,4,6]。

```

`思路`
无

`c# 实现`
```
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool IsInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     int GetInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     IList<NestedInteger> GetList();
 * }
 */
public class NestedIterator {
    List<int> mInteger;
    int index;
    public NestedIterator(IList<NestedInteger> nestedList) {
        index = 0;
        mInteger = new List<int>();
        mInteger = SplitNested(nestedList);
    }

    private List<int> SplitNested(IList<NestedInteger> nestedList){
        List<int> temp =  new List<int>();
        for (int i = 0; i < nestedList.Count; i++)
        {
            NestedInteger curr = nestedList[i];
            if (curr.IsInteger())
            {
                temp.Add(curr.GetInteger());
            }else{
                temp.AddRange(SplitNested(curr.GetList()));
            }
        }
        return temp;
    }

    public bool HasNext() {
        return index < mInteger.Count;
    }

    public int Next() {
        return mInteger[index++];
    }
}


/**
 * Your NestedIterator will be called like this:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.HasNext()) v[f()] = i.Next();
 */
```

***

## 2023/11/10

## 334. 递增的三元子序列

[334. 递增的三元子序列](https://leetcode.cn/problems/increasing-triplet-subsequence/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。

如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。

 

示例 1：

输入：nums = [1,2,3,4,5]
输出：true
解释：任何 i < j < k 的三元组都满足题意
示例 2：

输入：nums = [5,4,3,2,1]
输出：false
解释：不存在满足题意的三元组
示例 3：

输入：nums = [2,1,5,0,4,6]
输出：true
解释：三元组 (3, 4, 5) 满足题意，因为 nums[3] == 0 < nums[4] == 4 < nums[5] == 6
```

`思路`
双向遍历,用left right 两个辅助数组分别记录当前位置 i 从左到右和从右到左的最小值和最大值，如果存在一个nums[i] 满足大于 left[i] 并且小于 right[i] ，则符合三元递增

`c# 实现`
```
public class Solution {
    public bool IncreasingTriplet(int[] nums) {
        int n = nums.Length;
        int[] leftArr = new int[n] , rightArr = new int[n];
        leftArr[0] = nums[0];
        rightArr[n - 1] = nums[n - 1];
        for (int i = 1; i < n; i++)
        {
            leftArr[i] = Math.Min(leftArr[i - 1] , nums[i]);
        }
        for (int i = n - 2; i >= 0; i--)
        {
            rightArr[i] = Math.Max(rightArr[i + 1] , nums[i]);
        }
        for (int i = 1; i < n - 1; i++)
        {
            if (nums[i] > leftArr[i] && nums[i] < rightArr[i])
            {
                return true;
            }
        }
        return false;
    }
}
```

***

## 2023/11/9

## 329. 矩阵中的最长递增路径

[329. 矩阵中的最长递增路径](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

 

示例 1：


输入：matrix = [[9,9,4],[6,6,8],[2,1,1]]
输出：4 
解释：最长递增路径为 [1, 2, 6, 9]。
示例 2：


输入：matrix = [[3,4,5],[3,2,6],[2,2,1]]
输出：4 
解释：最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
示例 3：

输入：matrix = [[1]]
输出：1

```

`思路`
记忆优化递归

`c# 实现`
```
public class Solution {
    int[][] dp;
    public int LongestIncreasingPath(int[][] matrix) {
        int m = matrix.Length , n = matrix[0].Length;
        dp = new int[m][];
        for (int i = 0; i < m; i++)
        {
            dp[i] = new int[n];
        }
        int ans = 1;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                ans = Math.Max(GetCurrDp(matrix,i,j),ans);
            }
        }
        return ans;
    }

    private int GetCurrDp(int[][] matrix , int i , int j){
        if (dp[i][j] > 0)
        {
            return dp[i][j];
        }
        int l = 0, r =0, u = 0, d = 0;
        if (i - 1 >= 0 && i - 1 < matrix.Length && matrix[i][j] < matrix[i-1][j])
        {
            l = GetCurrDp(matrix,i-1,j);
        }
        if (i + 1 >= 0 && i + 1 < matrix.Length && matrix[i][j] < matrix[i+1][j])
        {
            r = GetCurrDp(matrix,i+1,j);
        }
        if (j - 1 >= 0 && j - 1 < matrix[0].Length && matrix[i][j] < matrix[i][j-1])
        {
            u = GetCurrDp(matrix,i,j-1);
        }
        if (j + 1 >= 0 && j + 1 < matrix[0].Length && matrix[i][j] < matrix[i][j+1])
        {
            d = GetCurrDp(matrix,i,j+1);
        }
        int maxNum = 1 + Math.Max(l, Math.Max(r, Math.Max(u,d)));
        dp[i][j] = maxNum;
        return maxNum;
    }
}
```

***

## 2023/11/8

## 328. 奇偶链表

[328. 奇偶链表](https://leetcode.cn/problems/odd-even-linked-list/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定单链表的头节点 head ，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。

第一个节点的索引被认为是 奇数 ， 第二个节点的索引为 偶数 ，以此类推。

请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。

你必须在 O(1) 的额外空间复杂度和 O(n) 的时间复杂度下解决这个问题。

 

示例 1:



输入: head = [1,2,3,4,5]
输出: [1,3,5,2,4]
示例 2:



输入: head = [2,1,3,5,6,4,7]
输出: [2,3,6,7,1,5,4]
```

`思路`
拆分后合并

`c# 实现`
```
public class Solution {
    public ListNode OddEvenList(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode evenHead = head.next;
        ListNode evenCurr = evenHead;
        ListNode curr = head;
        while(evenCurr != null && evenCurr.next != null){
            curr.next = evenCurr.next;
            if (evenCurr.next != null)
            {
                evenCurr.next = evenCurr.next.next;
            }
            curr = curr.next;
            evenCurr = evenCurr.next;
        }
        curr.next = evenHead;
        return head;
    }
}
```

***

## 2023/11/7

## 326. 3 的幂

[326. 3 的幂](https://leetcode.cn/problems/power-of-three/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 true ；否则，返回 false 。

整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3x

 

示例 1：

输入：n = 27
输出：true
示例 2：

输入：n = 0
输出：false
示例 3：

输入：n = 9
输出：true
示例 4：

输入：n = 45
输出：false

```

`思路`
除3判断余数

`c# 实现`
```
public class Solution {
    public bool IsPowerOfThree(int n) {
        while(n > 0 && n % 3 == 0){
            n /= 3;
        }
        return n == 1;
    }
}
```

***


## 2023/11/3

## 315. 计算右侧小于当前元素的个数

[315. 计算右侧小于当前元素的个数](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个整数数组 nums ，按要求返回一个新数组 counts 。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。

 

示例 1：

输入：nums = [5,2,6,1]
输出：[2,1,1,0] 
解释：
5 的右侧有 2 个更小的元素 (2 和 1)
2 的右侧仅有 1 个更小的元素 (1)
6 的右侧有 1 个更小的元素 (1)
1 的右侧有 0 个更小的元素
示例 2：

输入：nums = [-1]
输出：[0]
示例 3：

输入：nums = [-1,-1]
输出：[0,0]
```

`思路`
此题需要完全了解数组树的实现

`c# 实现`
```
public class Solution 
{
    private int[] c;

    private int[] a;

    private void Init(int length)
    {
        c = new int[length];
        Array.Fill(c, 0);
    }

    private int LowBit(int x)
    {
        return x & (-x);
    }

    private void Update(int pos)
    {
        while (pos < c.Length)
        {
            c[pos] += 1;
            pos += LowBit(pos);
        }
    }

    private int Query(int pos)
    {
        int ret = 0;
        while (pos > 0)
        {
            ret += c[pos];
            pos -= LowBit(pos);
        }

        return ret;
    }

    private void Discretization(int[] nums)
    {
        a = (int[])nums.Clone();
        var hashSet = new HashSet<int>(a);
        a = hashSet.ToArray();
        Array.Sort(a);
    }

    private int GetId(int x)
    {
        return Array.BinarySearch(a, x) + 1;
    }

    public IList<int> CountSmaller(int[] nums) 
    {
        var resultList = new List<int>(); 

        Discretization(nums);

        Init(nums.Length + 5);

        for (int i = nums.Length - 1; i >= 0; --i)
        {
            var id = GetId(nums[i]);
            resultList.Add(Query(id - 1));
            Update(id);
        }

        resultList.Reverse();

        return resultList;
    }
}

```

***

## 2023/11/2

## 295. 数据流的中位数

[295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

例如 arr = [2,3,4] 的中位数是 3 。
例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。
实现 MedianFinder 类:

MedianFinder() 初始化 MedianFinder 对象。

void addNum(int num) 将数据流中的整数 num 添加到数据结构中。

double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

示例 1：

输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

`思路`
优先队列

`c# 实现`
```
public class MedianFinder {
    PriorityQueue<int, int> minPri;
    PriorityQueue<int, int> maxPri;
    public MedianFinder() {
        minPri = new  PriorityQueue<int, int>();
        maxPri = new  PriorityQueue<int, int>();
    }
    
    public void AddNum(int num) {
        if (minPri.Count == 0 || num <= minPri.Peek())
        {
            minPri.Enqueue(num,-num);
            if (minPri.Count > maxPri.Count + 1)
            {
                int temp = minPri.Dequeue();
                maxPri.Enqueue(temp,temp);
            }
        }else{
            maxPri.Enqueue(num,num);
            if (maxPri.Count > minPri.Count)
            {
                int temp = maxPri.Dequeue();
                minPri.Enqueue(temp,-temp);
            }
        }
    }
    
    public double FindMedian() {
        if (minPri.Count > maxPri.Count)
        {
            return minPri.Peek();
        }else{
            return (minPri.Peek() + maxPri.Peek())/ 2.0;
        }
    }
}
```

***

## 2023/11/1

## 289. 生命游戏

[289. 生命游戏](https://leetcode.cn/problems/game-of-life/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
根据 百度百科 ， 生命游戏 ，简称为 生命 ，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。

给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态： 1 即为 活细胞 （live），或 0 即为 死细胞 （dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。给你 m x n 网格面板 board 的当前状态，返回下一个状态。

 

示例 1：


输入：board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
输出：[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
示例 2：


输入：board = [[1,1],[1,0]]
输出：[[1,1],[1,1]]

```

`思路`
标记需要修改的格子

`c# 实现`
```
public class Solution {
    int[][] around = new int[8][]{
        new int[]{-1,-1},new int[]{0,-1},new int[]{1,-1},
        new int[]{-1,0},new int[]{1,0},
        new int[]{-1,1},new int[]{0,1},new int[]{1,1}
    };
    public void GameOfLife(int[][] board) {
        int m = board.Length , n = board[0].Length;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                CountAlive(board,i,j);
            }
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (board[i][j] == 10)
                {
                    board[i][j] = 1;
                }else if (board[i][j] == 11)
                {
                    board[i][j] = 0;
                }
            }
        }
    }

    private void CountAlive(int[][] board , int ii, int ij){
        int count = 0 , m = board.Length , n = board[0].Length;
        for (int i = 0; i < around.Length; i++)
        {
            int x = ii + around[i][0];
            int y = ij + around[i][1];
            if (x < 0 || x >= m || y < 0 || y >= n)
            {
                continue;
            }
            count += (board[x][y] & 1);
        }
        if (board[ii][ij] == 0)
        {
            if (count == 3)
            {
                board[ii][ij] = 10;
            }
        }else{
            if (count < 2 || count > 3)
            {
                board[ii][ij] = 11;
            }
        }
    }
}
```

***

## 2023/10/31

## 268. 丢失的数字


[268. 丢失的数字](https://leetcode.cn/problems/missing-number/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。

 

示例 1：

输入：nums = [3,0,1]
输出：2
解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。
示例 2：

输入：nums = [0,1]
输出：2
解释：n = 2，因为有 2 个数字，所以所有的数字都在范围 [0,2] 内。2 是丢失的数字，因为它没有出现在 nums 中。
示例 3：

输入：nums = [9,6,4,2,3,5,7,0,1]
输出：8
解释：n = 9，因为有 9 个数字，所以所有的数字都在范围 [0,9] 内。8 是丢失的数字，因为它没有出现在 nums 中。
示例 4：

输入：nums = [0]
输出：1
解释：n = 1，因为有 1 个数字，所以所有的数字都在范围 [0,1] 内。1 是丢失的数字，因为它没有出现在 nums 中。
```

`思路`
排序

`c# 实现`
```
public class Solution {
    public int MissingNumber(int[] nums) {
        Array.Sort(nums);
        int missN = nums.Length;
        for (int i = 0; i < nums.Length; i++)
        {
            if (nums[i] != i)
            {
                missN = i;
                break;
            }
        }
        return missN;
    }
}
```

***

## 2023/10/30

## 242. 有效的字母异位词


[242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

 

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false

```

`思路`
哈希表记录

`c# 实现`
```
public class Solution {
    public bool IsAnagram(string s, string t) {
        if (s.Length != t.Length)
        {
            return false;
        }
        Dictionary<char,int> dict = new  Dictionary<char,int>();
        for (int i = 0; i < s.Length; i++)
        {
            dict.TryAdd(s[i],0);
            dict[s[i]] ++;
        }
        for (int i = 0; i < t.Length; i++)
        {
            if (dict.ContainsKey(t[i]))
            {
                dict[t[i]] --;
                if (dict[t[i]] == 0)
                {
                    dict.Remove(t[i]);
                }
            }else{
                return false;
            }
        }
        return true;
    }
}
```

***

## 2023/10/25

## 237. 删除链表中的节点


[237. 删除链表中的节点](https://leetcode.cn/problems/delete-node-in-a-linked-list/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
有一个单链表的 head，我们想删除它其中的一个节点 node。

给你一个需要删除的节点 node 。你将 无法访问 第一个节点  head。

链表的所有值都是 唯一的，并且保证给定的节点 node 不是链表中的最后一个节点。

删除给定的节点。注意，删除节点并不是指从内存中删除它。这里的意思是：

给定节点的值不应该存在于链表中。
链表中的节点数应该减少 1。
node 前面的所有值顺序相同。
node 后面的所有值顺序相同。
自定义测试：

对于输入，你应该提供整个链表 head 和要给出的节点 node。node 不应该是链表的最后一个节点，而应该是链表中的一个实际节点。
我们将构建链表，并将节点传递给你的函数。
输出将是调用你函数后的整个链表。
 

示例 1：


输入：head = [4,5,1,9], node = 5
输出：[4,1,9]
解释：指定链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9

```

`思路`
改变当前node的val,当作下个node

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public void DeleteNode(ListNode node) {
        while(node != null && node.next != null){
            node.val = node.next.val;
            if(node.next.next == null){
                node.next = null;
                break;
            }
            node = node.next;
        }
    }
}
```

***

## 2023/10/24

## 230. 二叉搜索树中第K小的元素


[230. 二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。

 

示例 1：


输入：root = [3,1,4,null,2], k = 1
输出：1
示例 2：


输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3

```

`思路`
中序遍历

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int KthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while(root != null || stack.Count > 0){
            while(root != null){
                stack.Push(root);
                root = root.left;
            }
            root = stack.Pop();
            k--;
            if(k == 0){
                break;
            }
            root = root.right;
        }
        return root.val;
    }
}
```

***

## 2023/10/20

## 227. 基本计算器 II


[227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

你可以假设给定的表达式总是有效的。所有中间结果将在 [-231, 231 - 1] 的范围内。

注意：不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。

 

示例 1：

输入：s = "3+2*2"
输出：7
示例 2：

输入：s = " 3/2 "
输出：1
示例 3：

输入：s = " 3+5 / 2 "
输出：5
 
```

`思路`
stack

`c# 实现`
```
public class Solution {
    public int Calculate(string s) {
        Stack<int> stack = new Stack<int>();
        char preSign = '+';
        int num = 0;
        int n = s.Length;
        for (int i = 0; i < n; ++i) {
            if (char.IsDigit(s[i])) {
                num = num * 10 + s[i] - '0';
            }
            if (!char.IsDigit(s[i]) && s[i] != ' ' || i == n - 1) {
                switch (preSign) {
                case '+':
                    stack.Push(num);
                    break;
                case '-':
                    stack.Push(-num);
                    break;
                case '*':
                    stack.Push(stack.Pop() * num);
                    break;
                default:
                    stack.Push(stack.Pop() / num);
                    break;
                }
                preSign = s[i];
                num = 0;
            }
        }
        int ans = 0;
        while (stack.Count > 0) {
            ans += stack.Pop();
        }
        return ans;
    }
}

```

***

## 2023/10/19

## 218. 天际线问题


[218. 天际线问题](https://leetcode.cn/problems/the-skyline-problem/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
城市的 天际线 是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。给你所有建筑物的位置和高度，请返回 由这些建筑物形成的 天际线 。

每个建筑物的几何信息由数组 buildings 表示，其中三元组 buildings[i] = [lefti, righti, heighti] 表示：

lefti 是第 i 座建筑物左边缘的 x 坐标。
righti 是第 i 座建筑物右边缘的 x 坐标。
heighti 是第 i 座建筑物的高度。
你可以假设所有的建筑都是完美的长方形，在高度为 0 的绝对平坦的表面上。

天际线 应该表示为由 “关键点” 组成的列表，格式 [[x1,y1],[x2,y2],...] ，并按 x 坐标 进行 排序 。关键点是水平线段的左端点。列表中最后一个点是最右侧建筑物的终点，y 坐标始终为 0 ，仅用于标记天际线的终点。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。

注意：输出天际线中不得有连续的相同高度的水平线。例如 [...[2 3], [4 5], [7 5], [11 5], [12 7]...] 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：[...[2 3], [4 5], [12 7], ...]

 

示例 1：


输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
解释：
图 A 显示输入的所有建筑物的位置和高度，
图 B 显示由这些建筑物形成的天际线。图 B 中的红点表示输出列表中的关键点。
示例 2：

输入：buildings = [[0,2,3],[2,5,3]]
输出：[[0,3],[5,0]]
```

`思路`
优先队列

`c# 实现`
```
public class Solution {
    public IList<IList<int>> GetSkyline(int[][] buildings) {
        IList<IList<int>> ans = new List<IList<int>>();
        List<int> boundaries = new List<int>();
        for (int i = 0; i < buildings.Length; i++)
        {
            boundaries.Add(buildings[i][0]);
            boundaries.Add(buildings[i][1]);
        }
        boundaries.Sort();
        PriorityQueue<int[] , int> priQueue = new PriorityQueue<int[], int>(new CompareMethod());
        int n = buildings.Length, idx = 0;
        foreach (int boundary in boundaries)
        {
            while (idx < n && buildings[idx][0] <= boundary)
            {
                priQueue.Enqueue(new int[2]{buildings[idx][1] , buildings[idx][2]} , buildings[idx][2]);
                idx++;
            }
            while (priQueue.Count > 0 && priQueue.Peek()[0] <= boundary) {
                priQueue.Dequeue();
            }
            int maxn = priQueue.Count == 0 ? 0 : priQueue.Peek()[1];
            if (ans.Count == 0 || maxn != ans[ans.Count - 1][1]) {
                ans.Add(new List<int>(){boundary,maxn});
            }
        }
        return ans;
    }
    public class CompareMethod : IComparer<int>  //继承IComparer<T>接口，T为要比较的元素的类型
    {                                             //类中类，也可以放在类外面
        public  int Compare(int x, int y)
        {
            return y - x;
        }
    }
}
```

***

## 2023/10/17

## 217. 存在重复元素


[217. 存在重复元素](https://leetcode.cn/problems/contains-duplicate/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。
 

示例 1：

输入：nums = [1,2,3,1]
输出：true
示例 2：

输入：nums = [1,2,3,4]
输出：false
示例 3：

输入：nums = [1,1,1,3,3,4,3,2,4,2]
输出：true
 
```

`思路`
排序

`c# 实现`
```
public class Solution {
    public bool ContainsDuplicate(int[] nums) {
        Array.Sort(nums);
        for (int i = 1; i < nums.Length; i++)
        {
            if (nums[i] == nums[i - 1])
            {
                return true;
            }
        }
        return false;
    }
}
```

***

## 2023/10/10

## 212. 单词搜索 II


[212. 单词搜索 II](https://leetcode.cn/problems/word-search-ii/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words， 返回所有二维网格上的单词 。

单词必须按照字母顺序，通过 相邻的单元格 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

 

示例 1：


输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
输出：["eat","oath"]
示例 2：


输入：board = [["a","b"],["c","d"]], words = ["abcb"]
输出：[]

```

`思路`
字典树

`c# 实现`
```
public class Solution {
      int[][] dirs = new int[][] {
        new int[]{1, 0},
        new int[]{-1, 0},
        new int[]{0, 1},
        new int[]{0, -1}
    };
    public IList<string> FindWords(char[][] board, string[] words) {
        Trie trie = new Trie();
        for (int i = 0; i < words.Length; i++)
        {
            string word = words[i];
            Trie curr = trie;
            for (int j = 0; j < word.Length; j++)
            {
                int mChar = word[j] - 'a';
                if (curr.Child[mChar] == null)
                {
                    curr.Child[mChar] = new Trie();
                }
                curr = curr.Child[mChar];
            }
            curr.Word = word;
        }
        ISet<string> ans = new HashSet<string>();
        for (int i = 0; i < board.Length; i++)
        {
            for (int j = 0; j < board[i].Length; j++)
            {
                DFS(board,trie,i,j,ans);
            }
        }
        return new List<string>(ans);
    }

    private void DFS(char[][] board , Trie curr, int i , int j , ISet<string> ans){
        char mChar = board[i][j];
        int mIndex = mChar - 'a';
        if (mIndex < 0 || mIndex > 25 || curr.Child[mIndex] == null)
        {
            return;
        }
        curr = curr.Child[mChar - 'a'];
        if (!"".Equals(curr.Word))
        {
            ans.Add(curr.Word);
        }
        board[i][j] = '#';
        for (int m = 0; m < dirs.Length; m++)
        {
            int mi = i + dirs[m][0];
            int mj = j + dirs[m][1];
            if (mi >= 0 && mj >= 0 && mi < board.Length && mj < board[0].Length)
            {
                DFS(board,curr,mi,mj,ans);
            }
        }
        board[i][j] = mChar;
    }

    class Trie {
        public string Word { get; set; }
        public Trie[] Child { get; set; }
        public Trie(){
            Word = "";
            Child = new Trie[26];
        }
    }
}
```

***

## 2023/10/9

## 210. 课程表 II


[210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。

 

示例 1：

输入：numCourses = 2, prerequisites = [[1,0]]
输出：[0,1]
解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
示例 2：

输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
输出：[0,2,1,3]
解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
示例 3：

输入：numCourses = 1, prerequisites = []
输出：[0]
```

`思路`
构建拓扑图

`c# 实现`
```
public class Solution {
    public int[] FindOrder(int numCourses, int[][] prerequisites) {
        List<int> ans = new List<int>();
        int[] inDegree = new int[numCourses];
        Array.Fill(inDegree,0);
        Dictionary<int,List<int>> require = new Dictionary<int,List<int>>();
        for (int i = 0; i < prerequisites.Length; i++)
        {
            inDegree[prerequisites[i][0]]++;
            require.TryAdd(prerequisites[i][1],new List<int>());
            require[prerequisites[i][1]].Add(prerequisites[i][0]);
        }
        Queue<int> que = new Queue<int>();
        for (int i = 0; i < inDegree.Length; i++)
        {
            if (inDegree[i] == 0)
            {
                que.Enqueue(i);
            }
        }
        while(que.Count > 0){
            int curr = que.Dequeue();
            ans.Add(curr);
            if (require.ContainsKey(curr))
            {
                List<int> reqList = require[curr];
                for (int i = 0; i < reqList.Count; i++)
                {
                    inDegree[reqList[i]] --;
                    if (inDegree[reqList[i]] == 0)
                    {
                        que.Enqueue(reqList[i]);
                    }
                }
            }
        }
        if (ans.Count == numCourses)
        {
            return ans.ToArray();
        }else{
            return new int[0];
        }
    }
}
```

***

## 2023/9/26

## 204. 计数质数


[204. 计数质数](https://leetcode.cn/problems/count-primes/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定整数 n ，返回 所有小于非负整数 n 的质数的数量 。

 

示例 1：

输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
示例 2：

输入：n = 0
输出：0
示例 3：

输入：n = 1
输出：0
```

`思路`
数组记录所有不是质数。

`c# 实现`
```
public class Solution {
    public int CountPrimes(int n) {
        int[] isPrime = new int[n];
        Array.Fill(isPrime, 1);
        int ans = 0;
        for (int i = 2; i < n; ++i) {
            if (isPrime[i] == 1) {
                ans += 1;
                if ((long) i * i < n) {
                    for (int j = i * i; j < n; j += i) {
                        isPrime[j] = 0;
                    }
                }
            }
        }
        return ans;
    }
}

```

***

## 2023/9/25

## 202. 快乐数


[202. 快乐数](https://leetcode.cn/problems/happy-number/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」 定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果这个过程 结果为 1，那么这个数就是快乐数。
如果 n 是 快乐数 就返回 true ；不是，则返回 false 。

 

示例 1：

输入：n = 19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
示例 2：

输入：n = 2
输出：false
```

`思路`
哈希记录循环

`c# 实现`
```
public class Solution {
    public bool IsHappy(int n) {
        HashSet<int> hset = new  HashSet<int>();
        while(n != 1){
            if (hset.Contains(n))
            {
                return false;
            }
            hset.Add(n);
            int sum = 0;
            while(n > 0){
                sum += (n % 10)*(n % 10);
                n /= 10;
            }
            n = sum;
        }
        return true;
    }
}
```

***


## 2023/9/21

## 191. 位1的个数


[191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

 

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在 示例 3 中，输入表示有符号整数 -3。
 

示例 1：

输入：n = 00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
示例 2：

输入：n = 00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
示例 3：

输入：n = 11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。

```

`思路`

`c# 实现`
```
public class Solution {
    public int HammingWeight(uint n) {
        int ans = 0;
        for (int i = 0; i < 32; i++)
        {
            if ((n & (1 << i)) != 0)
            {
                ans++;
            }
        }
        return ans;
    }
}
```

***


## 2023/9/20

## 190. 颠倒二进制位


[190. 颠倒二进制位](https://leetcode.cn/problems/reverse-bits/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
颠倒给定的 32 位无符号整数的二进制位。

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
 

示例 1：

输入：n = 00000010100101000001111010011100
输出：964176192 (00111001011110000010100101000000)
解释：输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
示例 2：

输入：n = 11111111111111111111111111111101
输出：3221225471 (10111111111111111111111111111111)
解释：输入的二进制串 11111111111111111111111111111101 表示无符号整数 4294967293，
     因此返回 3221225471 其二进制表示形式为 10111111111111111111111111111111 。

```

`思路`

`c# 实现`
```
public class Solution {
    public uint reverseBits(uint n) {
        uint revert = 0;
        for (int i = 0; i < 32 && n > 0; i++)
        {
            revert |= (n & 1) << (31 - i);
            n >>= 1;
        }
        return revert;
    }
}
```

***

## 2023/9/19

## 189. 轮转数组


[189. 轮转数组](https://leetcode.cn/problems/rotate-array/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

 

示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
示例 2:

输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

```

`思路`
循环修改

`c# 实现`
```
public class Solution {
    public void Rotate(int[] nums, int k) {
        int n = nums.Length;
        k %= n;
        if (n < 2 || k < 1)
        {
            return;
        }
        int start = 0 , next = 0;
        for (int i = 0; i < n; i++)
        {
            next = (next + k)%n;
            if (next == start)
            {
                start++;
                next = (next + 1)%n;
            }
            int temp = nums[next];
            nums[next] = nums[start];
            nums[start] = temp;
        }
    }
}
```

***

## 2023/9/15

## 179. 最大数


[179. 最大数](https://leetcode.cn/problems/largest-number/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。

 

示例 1：

输入：nums = [10,2]
输出："210"
示例 2：

输入：nums = [3,30,34,5,9]
输出："9534330"
```

`思路`
排序

`c# 实现`
```
public class Solution {
    public string LargestNumber(int[] nums) {
        Array.Sort(nums,(int a , int b)=>{
            long sa = 10 , sb = 10;
            while(sa <= a){
                sa *= 10;
            }while(sb <= b){
                sb *= 10;
            }
            return (int) ((sa * b + a) - (sb * a + b));
        });
        if (nums[0] == 0)
        {
            return "0";
        }
        string ans = "";
        for (int i = 0; i < nums.Length; i++)
        {
            ans += nums[i].ToString();
        }
        return ans;
    }
}
```

***

## 2023/9/14

## 172. 阶乘后的零

[172. 阶乘后的零](https://leetcode.cn/problems/factorial-trailing-zeroes/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一个整数 n ，返回 n! 结果中尾随零的数量。

提示 n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1

 

示例 1：

输入：n = 3
输出：0
解释：3! = 6 ，不含尾随 0
示例 2：

输入：n = 5
输出：1
解释：5! = 120 ，有一个尾随 0
示例 3：

输入：n = 0
输出：0

```

`思路`
数学。
实际上是计算有多少个质因数 5 或者 2 因为只有 2*5 = 10 而5一定不大于2 所以只要计算总共有多少个5

`c# 实现`
```
public class Solution {
    public int TrailingZeroes(int n) {
        int ans = 0;
        while(n > 0){
            n /= 5;
            ans += n;
        }
        return ans;
    }
}
```

***

## 2023/9/13

## 171. Excel 表列序号


[ 171. Excel 表列序号](https://leetcode.cn/problems/excel-sheet-column-number/description/?envType=featured-list&envId=2ckc81c?envType=featured-list&envId=2ckc81c)
```
给你一个字符串 columnTitle ，表示 Excel 表格中的列名称。返回 该列名称对应的列序号 。

例如：

A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...
 

示例 1:

输入: columnTitle = "A"
输出: 1
示例 2:

输入: columnTitle = "AB"
输出: 28
示例 3:

输入: columnTitle = "ZY"
输出: 701
```

`思路`

`c# 实现`
```
public class Solution {
    public int TitleToNumber(string columnTitle) {
        int n = columnTitle.Length , sum = 0;
        for (int i = 0; i < n; i++)
        {
            int unit = (int)Math.Pow(26,n - i - 1);
            int num = columnTitle[i] - 'A' + 1;
            sum += unit*num;
        }
        return sum;
    }
}
```

***

## 2023/9/12

## 166. 分数到小数


[166. 分数到小数](https://leetcode.cn/problems/fraction-to-recurring-decimal/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。

如果小数部分为循环小数，则将循环的部分括在括号内。

如果存在多个答案，只需返回 任意一个 。

对于所有给定的输入，保证 答案字符串的长度小于 104 。

 

示例 1：

输入：numerator = 1, denominator = 2
输出："0.5"
示例 2：

输入：numerator = 2, denominator = 1
输出："2"
示例 3：

输入：numerator = 4, denominator = 333
输出："0.(012)"
 

提示：

-231 <= numerator, denominator <= 231 - 1
denominator != 0
```

`思路`

`c# 实现`
```
public class Solution {
    public string FractionToDecimal(int numerator, int denominator) {
        long numeratorLong = (long) numerator;
        long denominatorLong = (long) denominator;
        if (numeratorLong % denominatorLong == 0)
        {
            return (numeratorLong / denominatorLong).ToString();
        }
        StringBuilder sb = new StringBuilder();
        if ((numeratorLong ^ denominatorLong) < 0 )
        {
            sb.Append('-');
        }

        numeratorLong = Math.Abs(numeratorLong);
        denominatorLong = Math.Abs(denominatorLong);
        long integerPart = numeratorLong / denominatorLong;
        sb.Append(integerPart);
        sb.Append('.');

        StringBuilder fractionPart = new StringBuilder();
        Dictionary<long, int> remainderIndexDic = new Dictionary<long, int>();
        long remainder = numeratorLong % denominatorLong;
        int index = 0;
        while(remainder != 0 &&  !remainderIndexDic.ContainsKey(remainder)){
            remainderIndexDic.Add(remainder,index++);
            remainder *= 10;
            fractionPart.Append(remainder / denominatorLong);
            remainder %= denominatorLong;
        }
        if (remainder != 0) { // 有循环节
            int insertIndex = remainderIndexDic[remainder];
            fractionPart.Insert(insertIndex, '(');
            fractionPart.Append(')');
        }
        sb.Append(fractionPart.ToString());

        return sb.ToString();
    }
}
```

***

## 2023/9/11

## 162. 寻找峰值


[162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

你必须实现时间复杂度为 O(log n) 的算法来解决此问题。

 

示例 1：

输入：nums = [1,2,3,1]
输出：2
解释：3 是峰值元素，你的函数应该返回其索引 2。
示例 2：

输入：nums = [1,2,1,3,5,6,4]
输出：1 或 5 
解释：你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。

```

`思路`
二分查找

`c# 实现`
```
public class Solution {
    public int FindPeakElement(int[] nums) {
        int left = 0, right = nums.Length - 1 , ans = -1;
        while(left <= right){
            int mid = (left + right)/2;
            if (Compare(nums,mid,mid -1) == 1 && Compare(nums,mid , mid + 1) == 1)
            {
                ans = mid;
                break;
            }
            if (Compare(nums,mid - 1,mid) == 1)
            {
                right = mid - 1;
            }else{
                left = mid + 1;
            }
        }
        return ans;
    }

    private int[] Get(int[] nums, int idx) {
        if (idx == -1 || idx == nums.Length) {
            return new int[]{0, 0};
        }
        return new int[]{1, nums[idx]};
    }

    private int Compare(int[] nums , int idx1 , int idx2){
        int[] num1 = Get(nums,idx1);
        int[] num2 = Get(nums,idx2);
        if (num1[0] != num2[0])
        {
            return num1[0] > num2[0] ? 1 : -1;
        }
        if (num1[1] == num2[1])
        {
            return 0;
        }
        return num1[1] > num2[1] ? 1 : -1;
    }
}
```

***

## 2023/9/8

## 150. 逆波兰表达式求值


[150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：

有效的算符为 '+'、'-'、'*' 和 '/' 。
每个操作数（运算对象）都可以是一个整数或者另一个表达式。
两个整数之间的除法总是 向零截断 。
表达式中不含除零运算。
输入是一个根据逆波兰表示法表示的算术表达式。
答案及所有中间计算结果可以用 32 位 整数表示。
 

示例 1：

输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
示例 2：

输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
示例 3：

输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

`思路`
遇到数字入栈，遇到运算 计算栈顶两个数

`c# 实现`
```
public class Solution {
    public int EvalRPN(string[] tokens) {
        Stack<int> stack = new  Stack<int>();
        for (int i = 0; i < tokens.Length; i++)
        {
            if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/" )
            {
                int num2 = stack.Pop();
                int num1 = stack.Pop();
                int res = 0;
                switch (tokens[i])
                {
                    case "+":
                        res = (num1 + num2);
                        break;
                    case "-":
                        res = (num1 - num2);
                        break;
                    case "*":
                        res = (num1 * num2);
                        break;
                    case "/":
                        res = (num1 / num2);
                        break;
                    default:
                        break;
                }
                stack.Push(res);
            }else{
                int num = int.Parse(tokens[i]);
                stack.Push(num);
            }
        }
        return stack.Pop();
    }
}

```

***

## 2023/9/6

## 140. 单词拆分 II


[140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。以任意顺序 返回所有这些可能的句子。

注意：词典中的同一个单词可能在分段中被重复使用多次。

 

示例 1：

输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
输出:["cats and dog","cat sand dog"]
示例 2：

输入:s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
输出:["pine apple pen apple","pineapple pen apple","pine applepen apple"]
解释: 注意你可以重复使用字典中的单词。
示例 3：

输入:s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
输出:[]

```

`思路`

`c# 实现`
```
public class Solution {
    public IList<string> WordBreak(string s, IList<string> wordDict) {
        Dictionary<int,List<List<string>>> map = new Dictionary<int,List<List<string>>>();
        List<List<string>> wordBreaks = Backtrack(s, s.Length, new HashSet<string>(wordDict), 0, map);
        IList<string> breakList = new List<string>();
        foreach (List<string> wordBreak in wordBreaks)
        {
             breakList.Add(string.Join(" ", wordBreak));
        }
        return breakList;
    }
    public List<List<string>> Backtrack(string s, int length, HashSet<string> wordSet, int index, Dictionary<int,List<List<string>>> map) {
        if (!map.ContainsKey(index)) {
            List<List<string>> wordBreaks = new List<List<string>>();
            if (index == length) {
                wordBreaks.Add(new List<string>());
            }
            for (int i = index + 1; i <= length; i++) {
                string word = s.Substring(index, i - index);
                if (wordSet.Contains(word)) {
                    List<List<string>> nextWordBreaks = Backtrack(s, length, wordSet, i, map);
                    foreach (List<string> nextWordBreak in nextWordBreaks)
                    {
                       List<string> wordBreak = new List<string>(nextWordBreak);
                        wordBreak.Insert(0,word);
                        wordBreaks.Add(wordBreak);
                    }
                }
            }
            map.Add(index,wordBreaks);
        }
        return map[index];
    }
}
```

***

## 2023/9/5

## 138. 复制带随机指针的链表


[138. 复制带随机指针的链表](https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

 

示例 1：



输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：



输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
示例 3：



输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

`思路`
哈希表递归

`c# 实现`
```
public class Solution {
    Dictionary<Node,Node> cacheNode = new Dictionary<Node,Node>();
    public Node CopyRandomList(Node head) {
        if (head == null)
        {
            return null;
        }
        if (!cacheNode.ContainsKey(head))
        {
            Node curr = new Node(head.val);
            cacheNode.Add(head,curr);
            curr.next = CopyRandomList(head.next);
            curr.random = CopyRandomList(head.random);
        }
        return cacheNode[head];
    }
}
```

***


## 2023/9/4

## 134. 加油站


[ 134. 加油站](https://leetcode.cn/problems/gas-station/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

给定两个整数数组 gas 和 cost ，如果你可以按顺序绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。

 

示例 1:

输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
示例 2:

输入: gas = [2,3,4], cost = [3,4,3]
输出: -1
解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。

```

`思路`
遍历

`c# 实现`
```
public class Solution {
    public int CanCompleteCircuit(int[] gas, int[] cost) {
        if (gas.Sum() < cost.Sum())
        {
            return -1;
        }
        int n = gas.Length , i = 0;
        while(i < n){
            int sumGas = 0 , sumCost = 0 , count = 0;
            while(count < n){
                int j = (count + i) % n;
                sumGas += gas[j];
                sumCost += cost[j];
                if (sumGas < sumCost)
                {
                    break;
                }
                count ++;
            }
            if (count == n){
                return i;
            }else{
                i = i + count + 1;
            }
        }
        return -1;
    }
}
```

***

## 2023/9/1

## 9. 回文数


[9. 回文数](https://leetcode.cn/problems/palindrome-number/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

例如，121 是回文，而 123 不是。

```

`思路`
反转一半的数字

`c# 实现`
```
public class Solution {
    public bool IsPalindrome(int x) {
        if(x < 0 || (x != 0 && x % 10 == 0)){
            return false;
        }
        int revert = 0;
        while(x > revert){
            revert = revert * 10 + x % 10;
            x /= 10;
        }
        return x == revert || x == revert / 10;
    }
}
```

***

## 2023/8/31

## 131. 分割回文串


[131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。

回文串 是正着读和反着读都一样的字符串。

 

示例 1：

输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
示例 2：

输入：s = "a"
输出：[["a"]]

```

`思路`


`c# 实现`
```
public class Solution {
    int[,] f;
    IList<IList<string>> ret; 
    IList<string> ans;
    int n;
    public IList<IList<string>> Partition(string s) {
        ret = new List<IList<string>>();
        ans = new List<string>();
        n = s.Length;
        f = new int[n,n];
        DFS(s, 0);
        return ret;
    }

    public void DFS(string s, int i) {
        if (i == n) {
            ret.Add(new List<string>(ans));
            return;
        }
        for (int j = i; j < n; ++j) {
            if (IsPalindrome(s, i, j) == 1) {
                ans.Add(s.Substring(i, j + 1 - i));
                DFS(s, j + 1);
                ans.RemoveAt(ans.Count - 1);
            }
        }
    }

    // 记忆化搜索中，f[i][j] = 0 表示未搜索，1 表示是回文串，-1 表示不是回文串
    public int IsPalindrome(string s, int i, int j) {
        if (f[i,j] != 0) {
            return f[i,j];
        }
        if (i >= j) {
            f[i,j] = 1;
        } else if (s[i] == s[j]) {
            f[i,j] = IsPalindrome(s, i + 1, j - 1);
        } else {
            f[i,j] = -1;
        }
        return f[i,j];
    }
}
```

***

## 2023/8/30

## 130. 被围绕的区域


[130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
 

示例 1：


输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
示例 2：

输入：board = [["X"]]
输出：[["X"]]

```

`思路`
DFS 将与边界相连标记，再遍历board 将为 O 的改为X 标记的还原为O

`c# 实现`
```
public class Solution {
    public void Solve(char[][] board) {
        int n = board.Length;
        int m = board[0].Length;
        for (int i = 0; i < n; i++)
        {
            DFS(board,i,0);
            DFS(board,i,m-1);
        }
        for (int i = 1; i < m; i++)
        {
            DFS(board,0,i);
            DFS(board,n - 1,i);
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (board[i][j] == 'V')
                {
                    board[i][j] = 'O';
                }else if(board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
            }
        }
    }

    private void DFS(char[][] board , int i , int j){
        if (i < 0 || i >= board.Length || j < 0 || j >= board[0].Length || board[i][j] != 'O')
        {
            return;
        }
        board[i][j] = 'V';
        DFS(board,i-1,j);
        DFS(board,i+1,j);
        DFS(board,i,j-1);
        DFS(board,i,j+1);
    }
}
```

***

## 2023/8/25

## 125. 验证回文串


[125. 验证回文串](https://leetcode.cn/problems/valid-palindrome/description/?envType=featured-list&envId=2ckc81c%3FenvType%3Dfeatured-list&envId=2ckc81c)
```
如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 回文串 。

字母和数字都属于字母数字字符。

给你一个字符串 s，如果它是 回文串 ，返回 true ；否则，返回 false 。

 

示例 1：

输入: s = "A man, a plan, a canal: Panama"
输出：true
解释："amanaplanacanalpanama" 是回文串。
示例 2：

输入：s = "race a car"
输出：false
解释："raceacar" 不是回文串。
示例 3：

输入：s = " "
输出：true
解释：在移除非字母数字字符之后，s 是一个空字符串 "" 。
由于空字符串正着反着读都一样，所以是回文串。

```

`思路`
双指针

`c# 实现`
```
public class Solution {
    public bool IsPalindrome(string s) {
        int l = 0 , r = s.Length - 1;
        while(l < r){
            while(l < r && !IsValid(s[l])){
                l++;
            }
            while(l < r && !IsValid(s[r])){
                r--;
            }
            if (l < r)
            {
                if (char.ToLower(s[l]) != char.ToLower(s[r]))
                {
                    return false;
                }
                l++;
                r--;
            }
         
        }
        return true;
    }

    private bool IsValid(char c){
        return char.IsLetter(c) || char.IsDigit(c);
    }
}
```

***

## 2023/8/24

## 122. 买卖股票的最佳时机 II


[122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/?envType=featured-list&envId=2ckc81c)
```
给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。

 

示例 1：

输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。
示例 2：

输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     总利润为 4 。
示例 3：

输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 交易无法获得正利润，所以不参与交易可以获得最大利润，最大利润为 0 。

```

`思路`
动态规划，其中dp[0,0] 表示当天没有股票的最大收益  dp[0,1]表示当天有股票的最大收益

`c# 实现`
```
public class Solution {
    public int MaxProfit(int[] prices) {
        int n = prices.Length;
        int[,] dp = new int[n,2];
        dp[0,1] = -prices[0];
        for (int i = 1; i < n; i++)
        {
            dp[i,0] = Math.Max(dp[i-1,0] , dp[i-1,1] + prices[i]);
            dp[i,1] = Math.Max(dp[i-1,1],dp[i-1,0] -prices[i]);
        }
        return dp[n-1,0];
    }
}
```

***

## 2023/8/23

## 118. 杨辉三角


[118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/description/?envType=featured-list&envId=2ckc81c)
```
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。



 

示例 1:

输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
示例 2:

输入: numRows = 1
输出: [[1]]

```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public IList<IList<int>> Generate(int numRows) {
        int[,] dp = new int[numRows+1,numRows + 1];
        dp[0,1] = 1;
        IList<IList<int>> ans = new List<IList<int>>();
        for (int i = 1; i <= numRows; i++)
        {
            List<int> list = new List<int>();
            for (int j = 1; j <= i; j++)
            {
                dp[i,j] = dp[i - 1,j - 1] + dp[i - 1 , j];
                list.Add(dp[i,j]);
            }
            ans.Add(list);
        }
        return ans;
    }
}
```

***

## 2023/8/22

## 116. 填充每个节点的下一个右侧节点指针


[116. 填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/description/?envType=featured-list&envId=2ckc81c)
```
给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

 

示例 1：



输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
示例 2:

输入：root = []
输出：[]
```

`思路`
BFS

`c# 实现`
```
public class Solution {
    public Node Connect(Node root) {
        if(root == null){
            return root;
        }
        Queue<Node> que = new Queue<Node>();
        que.Enqueue(root);
        while(que.Count > 0){
            int n = que.Count;
            for (int i = 0; i < n; i++)
            {
                 Node curr = que.Dequeue();
                 if (i < n-1)
                 {
                    curr.next = que.Peek();
                 }
                if (curr.left != null)
                {
                    que.Enqueue(curr.left);
                }
                if (curr.right != null)
                {
                    que.Enqueue(curr.right);
                }
            }
        }
        return root;
    }
}
```

***

## 2023/8/21

## 108. 将有序数组转换为二叉搜索树


[108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=featured-list&envId=2ckc81c)
```
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

 

示例 1：


输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

示例 2：


输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

`思路`
递归构造每个节点的left right

`c# 实现`
```
public class Solution {
    bool[] visited;
    public TreeNode SortedArrayToBST(int[] nums) {
        int n = nums.Length;
        visited = new bool[n];
        int mid  = n/2;
        visited[mid] = true;
        TreeNode root = new TreeNode(nums[mid]);
        AddNode(root,0 , mid , n - 1 , nums);
        return root;
    }

    private void AddNode(TreeNode root , int left , int mid , int right , int[] nums){
        int lmid = (left + mid ) / 2;
        int rmid = (mid + right +1) / 2;
        if(lmid < mid && !visited[lmid]){
            TreeNode leftNd = new TreeNode(nums[lmid]);
            root.left = leftNd;
            visited[lmid] = true;
            AddNode(leftNd ,left, lmid ,mid , nums);
        }
        if(rmid > mid && !visited[rmid]){
            TreeNode rightNd = new TreeNode(nums[rmid]);
            root.right = rightNd;
            visited[rmid] = true;
            AddNode(rightNd ,mid, rmid ,right , nums);
        }
    }
}
```

***

## 2023/8/18

## 103. 二叉树的锯齿形层序遍历


[103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/?envType=featured-list&envId=2ckc81c)
```
给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

 

示例 1：


输入：root = [3,9,20,null,null,15,7]
输出：[[3],[20,9],[15,7]]
示例 2：

输入：root = [1]
输出：[[1]]
示例 3：

输入：root = []
输出：[]
```

`思路`
BFS

`c# 实现`
```
public class Solution {
    public IList<IList<int>> ZigzagLevelOrder(TreeNode root) {
        IList<IList<int>> ans = new List<IList<int>>();
        if (root == null)
        {
            return ans;
        }
        Queue<TreeNode> que = new Queue<TreeNode>();
        que.Enqueue(root);
        bool isLeftOrder = true;
        while(que.Count > 0){
            int n = que.Count;
            IList<int> temp = new List<int>();
            for (int i = 0; i < n; i++)
            {
                TreeNode curr = que.Dequeue();
                if (isLeftOrder)
                {
                    temp.Add(curr.val);
                }else{
                    temp.Insert(0,curr.val);
                }
                if (curr.left != null)
                {
                    que.Enqueue(curr.left);
                }
                if (curr.right != null)
                {
                    que.Enqueue(curr.right);
                }
            }
            isLeftOrder = !isLeftOrder;
            ans.Add(temp);
        }
        return ans;
    }
}
```

***

## 2023/8/17

## 91. 解码方法


[91. 解码方法](https://leetcode.cn/problems/decode-ways/description/?envType=featured-list&envId=2ckc81c)
```
一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

"AAJF" ，将消息分组为 (1 1 10 6)
"KJF" ，将消息分组为 (11 10 6)
注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

题目数据保证答案肯定是一个 32 位 的整数。

 

示例 1：

输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2：

输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
示例 3：

输入：s = "06"
输出：0
解释："06" 无法映射到 "F" ，因为存在前导零（"6" 和 "06" 并不等价）。
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public int NumDecodings(string s) {
        int n = s.Length;
        int[] dp = new int[n+1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++)
        {
            if (s[i - 1] != '0')
            {
                dp[i] += dp[i - 1];
            }
            if (i > 1 && s[i - 2] != '0' && ( (s[i - 2] - '0')*10 + (s[i - 1] - '0') < 27))
            {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }
}
```

***

## 2023/8/16

## 88. 合并两个有序数组


[88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/description/?envType=featured-list&envId=2ckc81c)
```
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。

 

示例 1：

输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。
示例 2：

输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
解释：需要合并 [1] 和 [] 。
合并结果是 [1] 。
示例 3：

输入：nums1 = [0], m = 0, nums2 = [1], n = 1
输出：[1]
解释：需要合并的数组是 [] 和 [1] 。
合并结果是 [1] 。
注意，因为 m = 0 ，所以 nums1 中没有元素。nums1 中仅存的 0 仅仅是为了确保合并结果可以顺利存放到 nums1 中。
```

`思路`
3指针遍历 

`c# 实现`
```
public class Solution {
    public void Merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1 , j = n - 1 , k = m + n - 1;
        while(k >= 0 && j >= 0 && i >= 0){
            nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
        }
        if (j >= 0)
        {
            for (int l = 0; l <= j; l++)
            {
                nums1[l] = nums2[l];
            }
        }
    }
}
```

***

## 2023/8/15

## 73. 矩阵置零


[73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=featured-list&envId=2ckc81c)
```
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

 

示例 1：


输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
示例 2：


输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]

```

`思路`
标记遍历两次

`c# 实现`
```
public class Solution {
    public void SetZeroes(int[][] matrix) {
        int m = matrix.Length, n = matrix[0].Length;
        bool[] row = new bool[m];
        bool[] col = new bool[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = col[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}

```

***

## 2023/8/14

## 69. x 的平方根


[69. x 的平方根](https://leetcode.cn/problems/sqrtx/description/?envType=featured-list&envId=2ckc81c)
```
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。

 

示例 1：

输入：x = 4
输出：2
示例 2：

输入：x = 8
输出：2
解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
```

`思路`
二分查找

`c# 实现`
```
public class Solution {
    public int MySqrt(int x) {
        int l = 0 , r = x , ans = -1;
        while(l <= r){
            int mid =  (r + l) / 2;
            if ((long)mid * mid <= x)
            {
                ans = mid;
                l = mid + 1;
            }else{
                r = mid - 1;
            }
        }
        return ans;
    }
}
```

***

## 2023/8/11

## 66. 加一


[66. 加一](https://leetcode.cn/problems/plus-one/description/?envType=featured-list&envId=2ckc81c)
```
给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

 

示例 1：

输入：digits = [1,2,3]
输出：[1,2,4]
解释：输入数组表示数字 123。
示例 2：

输入：digits = [4,3,2,1]
输出：[4,3,2,2]
解释：输入数组表示数字 4321。
示例 3：

输入：digits = [0]
输出：[1]
```

`思路`
倒序便利添加

`c# 实现`
```
public class Solution {
    public int[] PlusOne(int[] digits) {
        List<int> ans = new List<int>();
        int n = digits.Length;
        bool addone = true;
        for (int i = n - 1; i >= 0; i--)
        {
            int dig = digits[i];
            if (addone)
            {
                dig++;
                if (dig > 9)
                {
                    ans.Insert(0,0);
                    if (i == 0)
                    {
                        ans.Insert(0,1);
                    }
                }else{
                    addone = false;
                    ans.Insert(0,dig);
                }
            }else{
                ans.Insert(0,dig);
            }
        }
        return ans.ToArray();
    }
}
```

***

## 2023/8/10

## 54. 螺旋矩阵


[54. 螺旋矩阵](https://github.com/h87545645/leetcode_note/edit/main/README.md)
```
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

 

示例 1：


输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：


输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

`思路`
模拟

`c# 实现`
```
public class Solution {
    public IList<int> SpiralOrder(int[][] matrix) {
        IList<int> ans = new List<int>();
        int[][] directions = new int[4][]{
            new int[]{0, 1}, new int[]{1, 0}, new int[]{0, -1}, new int[]{-1, 0}};
        int directionIndex = 0;
        int n = matrix.Length;
        int m = matrix[0].Length;
        int total = n * m;
        int row = 0 ,col = 0;
        for (int i = 0; i < total; i++)
        {
            ans.Add(matrix[row][col]);
            matrix[row][col] = int.MinValue;
            int nextRow = row + directions[directionIndex][0];
            int nextCol = col + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= n || nextCol < 0 || nextCol >= m || matrix[nextRow][nextCol] == int.MinValue)
            {
               directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            col += directions[directionIndex][1];
        }
        return ans;
    }
}
```

***

## 2023/8/9

## 50. Pow(x, n)


[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/description/?envType=featured-list&envId=2ckc81c)
```
实现 pow(x, n) ，即计算 x 的整数 n 次幂函数（即，xn ）。

 

示例 1：

输入：x = 2.00000, n = 10
输出：1024.00000
示例 2：

输入：x = 2.10000, n = 3
输出：9.26100
示例 3：

输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

`思路`
快速幂

`c# 实现`
```
public class Solution {
    public double MyPow(double x, int n) {
        return n >= 0 ? QuickMul(x,n) : 1.0 / QuickMul(x, - n);
    }

    private double QuickMul(double x, int n){
        if (n == 0)
        {
            return 1.0;
        }
        double y = QuickMul(x,n/2);
        return (n&1) == 0 ? y*y : y*y*x;
    }
}
```

***

## 2023/8/8

## 44. 通配符匹配


[44. 通配符匹配](https://leetcode.cn/problems/wildcard-matching/description/?envType=featured-list&envId=2ckc81c)
```
给你一个输入字符串 (s) 和一个字符模式 (p) ，请你实现一个支持 '?' 和 '*' 匹配规则的通配符匹配：
'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符序列（包括空字符序列）。
判定匹配成功的充要条件是：字符模式必须能够 完全匹配 输入字符串（而不是部分匹配）。

 
示例 1：

输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
示例 2：

输入：s = "aa", p = "*"
输出：true
解释：'*' 可以匹配任意字符串。
示例 3：

输入：s = "cb", p = "?a"
输出：false
解释：'?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public bool IsMatch(string s, string p) {
        int m = s.Length, n = p.Length;
        bool[][] dp = new bool[m + 1][];
        for (int i = 0; i <= m; i++)
        {
            dp[i] = new bool[n + 1];
        }
        dp[0][0] = true;
        for (int i = 1; i <= n; i++)
        {
            if (p[i - 1] == '*')
            {
                dp[0][i] = true;
            }
            else{
                break;
            }
        }
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                if (p[j - 1] == '*')
                {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }else if(p[j - 1] == '?' || p[j - 1] == s[i - 1]){
                    dp[i][j] = dp[i - 1][j -1];
                }
            }
        }
        return dp[m][n];
    }
}
```

***

## 2023/8/7

## 41. 缺失的第一个正数


[41. 缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/description/?envType=featured-list&envId=2ckc81c)
```
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
 

示例 1：

输入：nums = [1,2,0]
输出：3
示例 2：

输入：nums = [3,4,-1,1]
输出：2
示例 3：

输入：nums = [7,8,9,11,12]
输出：1
```

`思路`
原地哈希

`c# 实现`
```
public class Solution {
    public int FirstMissingPositive(int[] nums) {
        int n = nums.Length;
        for (int i = 0; i < n; i++)
        {
            while(nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]){
                int temp = nums[nums[i] - 1];
                 nums[nums[i] - 1] = nums[i];
                 nums[i] = temp;
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }
}
```

***

## 2023/8/4

## 38. 外观数列

[38. 外观数列](https://leetcode.cn/problems/count-and-say/description/?envType=featured-list&envId=2ckc81c)
```
给定一个正整数 n ，输出外观数列的第 n 项。

「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

你可以将其视作是由递归公式定义的数字字符串序列：

countAndSay(1) = "1"
countAndSay(n) 是对 countAndSay(n-1) 的描述，然后转换成另一个数字字符串。
前五项如下：

1.     1
2.     11
3.     21
4.     1211
5.     111221
第一项是数字 1 
描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
要 描述 一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多 相同字符 组成。然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。

例如，数字字符串 "3322251" 的描述如下图：


 

示例 1：

输入：n = 1
输出："1"
解释：这是一个基本样例。
示例 2：

输入：n = 4
输出："1211"
解释：
countAndSay(1) = "1"
countAndSay(2) = 读 "1" = 一 个 1 = "11"
countAndSay(3) = 读 "11" = 二 个 1 = "21"
countAndSay(4) = 读 "21" = 一 个 2 + 一 个 1 = "12" + "11" = "1211"
```

`思路`
遍历

`c# 实现`
```
public class Solution {
    public string CountAndSay(int n) {
        string prev = "1";
        for (int i = 1; i < n; i++)
        {
            string curr = "";
            int start = 0, pos = 0;
            while(pos < prev.Length){
                while(pos < prev.Length && prev[start] == prev[pos]){
                    pos++;
                }
                int cnt = pos - start;
                curr += (cnt.ToString() + prev[start]);
                start = pos;
            }
            prev = curr;
        }
        return prev;
    }
}
```

***

## 2023/8/3

## 36. 有效的数独

[36. 有效的数独](https://leetcode.cn/problems/valid-sudoku/description/?envType=featured-list&envId=2ckc81c)
```
请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
 

注意：

一个有效的数独（部分已被填充）不一定是可解的。
只需要根据以上规则，验证已经填入的数字是否有效即可。
空白格用 '.' 表示。
 

示例 1：


输入：board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：true
示例 2：

输入：board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
输出：false
解释：除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。 但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。

```

`思路`
分别用三个数组记录行列块数字出现次数

`c# 实现`
```
public class Solution {
    public bool IsValidSudoku(char[][] board) {
        int[,] rows = new int[9,9];
        int[,] columns = new int[9,9];
        int [, ,] box = new int[3 , 3 , 9];
        for (int i = 0; i < board.Length; i++)
        {
            for (int j = 0; j < board[0].Length; j++)
            {
                if (board[i][j] == '.')
                {
                    continue;
                }
                int val = board[i][j] - '0' - 1;
                if (rows[i,val] > 0)
                {
                    return false;
                }
                if (columns[j,val] > 0)
                {
                    return false;
                }
                int temprow = i / 3;
                int tempcol = j / 3;
                if (box[temprow , tempcol , val] > 0)
                {
                    return false;
                }
                rows[i,val] ++;
                columns[j,val] ++;
                box[temprow , tempcol , val] ++;
            }
        }
        return true;
    }
}
```

***

## 2023/8/1

## 26. 删除有序数组中的重复项


[26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/description/?envType=featured-list&envId=2ckc81c)
```
给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。

考虑 nums 的唯一元素的数量为 k ，你需要做以下事情确保你的题解可以被通过：

更改数组 nums ，使 nums 的前 k 个元素包含唯一元素，并按照它们最初在 nums 中出现的顺序排列。nums 的其余元素与 nums 的大小不重要。
返回 k 。
判题标准:

系统会用下面的代码来测试你的题解:

int[] nums = [...]; // 输入数组
int[] expectedNums = [...]; // 长度正确的期望答案

int k = removeDuplicates(nums); // 调用

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
如果所有断言都通过，那么您的题解将被 通过。

 

示例 1：

输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
示例 2：

输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
```

`思路`
双指针

`c# 实现`
```
public class Solution {
    public int RemoveDuplicates(int[] nums) {
        int n = nums.Length;
        int i = 0, j = 1;
        while(j < n){
            if (nums[i] != nums[j])
            {
                nums[++i] = nums[j];
            }
            j++;
        }
        return i + 1;
    }
}
```

***

## 2023/7/31

## 14. 最长公共前缀


[14. 最长公共前缀](https://leetcode.cn/problems/longest-common-prefix/description/?envType=featured-list&envId=2ckc81c)
```
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

 

示例 1：

输入：strs = ["flower","flow","flight"]
输出："fl"
示例 2：

输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。
```

`思路`

`c# 实现`
```
public class Solution {
    public string LongestCommonPrefix(string[] strs) {
        int n = int.MaxValue;
        for (int i = 0; i < strs.Length; i++)
        {
            n = Math.Min(strs[i].Length,n);
        }
        string ans = "";
        for (int i = 0; i < n; i++)
        {
            char temp = strs[0][i];
            for (int j = 1; j < strs.Length; j++)
            {
                if (strs[j][i] != temp)
                {
                    return ans;
                }
            }
            ans += temp;
        }
        return ans;
    }
}
```

***

## 2023/7/28

## 13. 罗马数字转整数


[13. 罗马数字转整数](https://leetcode.cn/problems/roman-to-integer/description/?envType=featured-list&envId=2ckc81c)
```
罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。

 

示例 1:

输入: s = "III"
输出: 3
示例 2:

输入: s = "IV"
输出: 4
示例 3:

输入: s = "IX"
输出: 9
示例 4:

输入: s = "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
示例 5:

输入: s = "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

`思路`

`c# 实现`
```
public class Solution {
    public int RomanToInt(string s) {
        int n = s.Length;
        int ans = 0;
        int pre = GetValue(s[0]);
        for (int i = 1; i < n; i++)
        {
            int curr = GetValue(s[i]);
            if (pre < curr)
            {
                ans -= pre;
            }else{
                ans += pre;
            }
            pre = curr;
        }
        ans += pre;
        return ans;
    }

    private int GetValue(char c){
        switch (c)
        {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
                
            default:
                return 0;
        }
    }
}
```

***

## 2023/7/27

## 8. 字符串转换整数 (atoi)


[8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/description/?envType=featured-list&envId=2ckc81c)
```
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
 

示例 1：

输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
第 1 步："42"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："42"（读入 "42"）
           ^
解析得到整数 42 。
由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
示例 2：

输入：s = "   -42"
输出：-42
解释：
第 1 步："   -42"（读入前导空格，但忽视掉）
            ^
第 2 步："   -42"（读入 '-' 字符，所以结果应该是负数）
             ^
第 3 步："   -42"（读入 "42"）
               ^
解析得到整数 -42 。
由于 "-42" 在范围 [-231, 231 - 1] 内，最终结果为 -42 。
示例 3：

输入：s = "4193 with words"
输出：4193
解释：
第 1 步："4193 with words"（当前没有读入字符，因为没有前导空格）
         ^
第 2 步："4193 with words"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
         ^
第 3 步："4193 with words"（读入 "4193"；由于下一个字符不是一个数字，所以读入停止）
             ^
解析得到整数 4193 。
由于 "4193" 在范围 [-231, 231 - 1] 内，最终结果为 4193 。
```

`思路`

`c# 实现`
```
public class Solution {
    public int MyAtoi(string s) {
        int n = s.Length;
        if (n == 0)
        {
            return 0;
        }
        int ans = 0;
        int sign = 1;
        int index = 0;
        while(index < n && s[index] == ' '){
            index++;
        }
        while(index < n && (s[index] == '+' || s[index] == '-')){
            sign = s[index++] == '-' ? -1 : 1;
            break;
        }
        while(index < n && char.IsDigit(s[index])){
            int digit = s[index] - '0';
            if (ans > (int.MaxValue - digit)/10)
            {
                return sign == 1 ? int.MaxValue : int.MinValue;
            }
            ans = ans*10 + digit;
            index++;
        }
        return ans*sign;
    }
}
```

***

## 2023/7/26

## 739. 每日温度

[739. 每日温度](https://leetcode.cn/problems/daily-temperatures/description/?envType=featured-list&envId=2cktkvj)
```
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

 

示例 1:

输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
示例 2:

输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
示例 3:

输入: temperatures = [30,60,90]
输出: [1,1,0]
```

`思路`
递减栈

`c# 实现`
```
public class Solution {
    public int[] DailyTemperatures(int[] temperatures) {
        int n = temperatures.Length;
        int[] ans = new int[n];
        Stack<int> st = new Stack<int>();
        for (int i = 0; i < n; i++)
        {
            while(st.Count > 0 && temperatures[i] > temperatures[st.Peek()]){
                int temp = st.Pop();
                ans[temp] = i - temp;
            }
            st.Push(i);
        }
        return ans;
    }
}
```

***

## 2023/7/25

## 647. 回文子串


[647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/description/?envType=featured-list&envId=2cktkvj)
```
给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。

回文字符串 是正着读和倒过来读一样的字符串。

子字符串 是字符串中的由连续字符组成的一个序列。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

 

示例 1：

输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
示例 2：

输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

`思路`
中心拓展

`c# 实现`
```
public class Solution {
    public int CountSubstrings(string s) {
        int n = s.Length;
        int ans = 0;
        for (int i = 0; i < 2*n - 1; i++)
        {
            int l = i/2;
            int r = i/2 + i%2;
            while(l >= 0 && r < n && s[l] == s[r]){
                ans++;
                l--;
                r++;
            }
        }
        return ans;
    }
}
```

***

## 2023/7/24

## 621. 任务调度器


[621. 任务调度器](https://leetcode.cn/problems/task-scheduler/)
```
给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。

然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

你需要计算完成所有任务所需要的 最短时间 。

 

示例 1：

输入：tasks = ["A","A","A","B","B","B"], n = 2
输出：8
解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B
     在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。 
示例 2：

输入：tasks = ["A","A","A","B","B","B"], n = 0
输出：6
解释：在这种情况下，任何大小为 6 的排列都可以满足要求，因为 n = 0
["A","A","A","B","B","B"]
["A","B","A","B","A","B"]
["B","B","B","A","A","A"]
...
诸如此类
示例 3：

输入：tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
输出：16
解释：一种可能的解决方案是：
     A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> (待命) -> (待命) -> A -> (待命) -> (待命) -> A
```

`思路`
模拟

`c# 实现`
```
public class Solution {
    public int LeastInterval(char[] tasks, int n) {
        int max = 0;
        int maxCnt = 0;
        int[] temp = new int[26];
        for (int i = 0; i < tasks.Length; i++)
        {
            char c = tasks[i];
            temp[c - 'A']++;
            max = Math.Max(temp[c - 'A'] , max);
        }
        for (int i = 0; i < 26; i++)
        {
            if (temp[i] == max)
            {
               maxCnt ++;
            }
        }
        return Math.Max(tasks.Length , (max - 1)*(n + 1) + maxCnt);
    }
}
```

***

## 2023/7/21

## 617. 合并二叉树


[617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/description/?envType=featured-list&envId=2cktkvj)
```
给你两棵二叉树： root1 和 root2 。

想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；否则，不为 null 的节点将直接作为新二叉树的节点。

返回合并后的二叉树。

注意: 合并过程必须从两个树的根节点开始。

 

示例 1：


输入：root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
输出：[3,4,5,5,4,null,7]
示例 2：

输入：root1 = [1], root2 = [1,2]
输出：[2,2]
```

`思路`
DFS

`c# 实现`
```
public class Solution {
    TreeNode ans;
    public TreeNode MergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
        {
            return null;
        }
        int val = (root1 != null ? root1.val : 0) + (root2 != null ? root2.val : 0);
        TreeNode ans = new TreeNode(val);
        ans.left = MergeTrees(root1?.left,root2?.left);
        ans.right = MergeTrees(root1?.right,root2?.right);
        return ans;
    }
}
```

***

## 2023/7/20

## 581. 最短无序连续子数组

[581. 最短无序连续子数组](https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/description/?envType=featured-list&envId=2cktkvj)
```
给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

请你找出符合题意的 最短 子数组，并输出它的长度。

 

示例 1：

输入：nums = [2,6,4,8,10,9,15]
输出：5
解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
示例 2：

输入：nums = [1,2,3,4]
输出：0
示例 3：

输入：nums = [1]
输出：0
```

`思路`
双指针遍历

`c# 实现`
```
public class Solution {
    public int FindUnsortedSubarray(int[] nums) {
        int n = nums.Length;
        int left = -1, right = -2;
        int maxn = int.MinValue;
        int minn = int.MaxValue;
        for (int i = 0; i < n; i++)
        {
           if (maxn > nums[i])
           {
            right = i;
           }else{
            maxn = nums[i]; 
           }
           if (minn < nums[n - i - 1])
           {
            left = n - i - 1;
           }else{
            minn = nums[n - i - 1];
           }
        }
        return right - left + 1;
    }
}
```

***


## 2023/7/19

## 560. 和为 K 的子数组

[560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=featured-list&envId=2cktkvj)
```
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。

 

示例 1：

输入：nums = [1,1,1], k = 2
输出：2
示例 2：

输入：nums = [1,2,3], k = 3
输出：2
```

`思路`
前缀和

`c# 实现`
```
public class Solution {
    public int SubarraySum(int[] nums, int k) {
        int n = nums.Length;
        int ans = 0;
        int pre = 0;
        Dictionary<int,int> dict = new Dictionary<int,int>();
        dict.Add(0,1);
        for (int i = 0; i < n; i++)
        {
            pre += nums[i];
            int temp = pre - k;
            if (dict.ContainsKey(temp))
            {
                ans += dict[temp];
            }
            dict.TryAdd(pre,0);
            dict[pre] ++;
        }
        return ans;
    }
}
```

***

## 2023/7/18

## 543. 二叉树的直径

[543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=featured-list&envId=2cktkvj)
```
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。

 

示例 1：


输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
示例 2：

输入：root = [1,2]
输出：1

```

`思路`
DFS二叉树，记录最长路径max

`c# 实现`
```
public class Solution {
    private int max;
    public int DiameterOfBinaryTree(TreeNode root) {
        max = 0;
        DFS(root);
        return max - 1;
    }
    
    private int DFS(TreeNode root){
        if (root == null)
        {
            return 0;
        }
        int left = DFS(root.left);
        int right = DFS(root.right);
        max = Math.Max(max , left + right + 1);
        return Math.Max(left + 1, right + 1);
    }
}
```

***

## 2023/7/17

## 538. 把二叉搜索树转换为累加树


[538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/description/?envType=featured-list&envId=2cktkvj)
```
给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。
注意：本题和 1038: https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/ 相同

 

示例 1：



输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
示例 2：

输入：root = [0,null,1]
输出：[1,null,1]
示例 3：

输入：root = [1,0,2]
输出：[3,3,2]
示例 4：

输入：root = [3,2,4,1]
输出：[7,9,4,10]
```

`思路`
后序遍历二叉树赋值

`c# 实现`
```
public class Solution {
    private int sum;
    public TreeNode ConvertBST(TreeNode root) {
        sum = 0;
        DFS(root);
        return root;
    }

    private void DFS(TreeNode root){
        if (root == null)
        {
            return;
        }
        DFS(root.right);
        sum += root.val;
        root.val = sum;
        DFS(root.left);
    }
}
```

***

## 2023/7/14

## 494. 目标和

[494. 目标和](https://leetcode.cn/problems/target-sum/description/?envType=featured-list&envId=2cktkvj)
```
给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

 

示例 1：

输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
示例 2：

输入：nums = [1], target = 1
输出：1
```

`思路`

`c# 实现`
```
public class Solution {
    int count = 0;

    public int FindTargetSumWays(int[] nums, int target) {
        Search(nums,target,0,0);
        return count;
    }

    private void Search(int[] nums, int target , int index , int sum){
        if (index == nums.Length)
        {
            if (sum == target)
            {
                count++;
            }
        }else{
            Search(nums,target,index+1,sum + nums[index]);
            Search(nums,target,index+1,sum - nums[index]);
        }
    }
  
}
```

***

## 2023/7/13

## 461. 汉明距离

[461. 汉明距离](https://leetcode.cn/problems/hamming-distance/description/?envType=featured-list&envId=2cktkvj)
```
两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。

给你两个整数 x 和 y，计算并返回它们之间的汉明距离。

 

示例 1：

输入：x = 1, y = 4
输出：2
解释：
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
上面的箭头指出了对应二进制位不同的位置。
示例 2：

输入：x = 3, y = 1
输出：1
```

`思路`
每位与上1判断是否相同

`c# 实现`
```
public class Solution {
    public int HammingDistance(int x, int y) {
        int ans = 0;
        while(x > 0 || y > 0){
            if ((x & 1) != (y & 1))
            {
                ans++;
            }
            x >>= 1;
            y >>= 1;
        }
        return ans;
    }
}
```

***

## 2023/7/12

## 448. 找到所有数组中消失的数字

[448. 找到所有数组中消失的数字](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/description/?envType=featured-list&envId=2cktkvj)
```
给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。

 

示例 1：

输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6]
示例 2：

输入：nums = [1,1]
输出：[2]
```

`思路`
双指针

`c# 实现`
```
public class Solution {
    public IList<int> FindDisappearedNumbers(int[] nums) {
        int n = nums.Length;
        IList<int> ans = new List<int>();
        Array.Sort(nums);
        for (int i = 0 , j = 1 ; j <= n; j++)
        {
            while(nums[i] < j && i < n - 1){
                i++;
            }
            if (nums[i] > j)
            {
                ans.Add(j);
            }else if(nums[i] < j){
                for (; j <= n; j++)
                {
                    ans.Add(j);
                }
                break;
            }
        }
        return ans;
    }
}
```

***

## 2023/7/11

## 438. 找到字符串中所有字母异位词

[438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=featured-list&envId=2cktkvj)
```

```

`思路`
滑动窗口

`c# 实现`
```
public class Solution {
    public IList<int> FindAnagrams(string s, string p) {
        int sLen = s.Length, pLen = p.Length;
        if (sLen < pLen) {
            return new List<int>();
        }
        IList<int> ans = new List<int>();
        int[] sCount = new int[26];
        int[] pCount = new int[26];
        for (int i = 0; i < pLen; i++)
        {
            ++sCount[s[i] - 'a'];
            ++pCount[p[i] - 'a'];
        }
        if (Enumerable.SequenceEqual(sCount, pCount)) {
            ans.Add(0);
        }
        for (int i = 0; i < sLen - pLen; i++)
        {
            --sCount[s[i]-'a'];
            ++sCount[s[i+pLen] - 'a'];
             if (Enumerable.SequenceEqual(sCount, pCount)) {
                ans.Add(i+1);
            }
        }
        return ans;
    }
}

```

***

## 2023/7/10

## 437. 路径总和 III

[437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/description/)
```
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

 

示例 1：



输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
示例 2：

输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3

```

`思路`
前缀和

`c# 实现`
```
public class Solution {
    public int PathSum(TreeNode root, int targetSum) {
        Dictionary<long, int> prefix = new Dictionary<long, int>();
        prefix.Add(0, 1);
        return DFS(root, prefix, 0, targetSum);
    }

    public int DFS(TreeNode root, Dictionary<long, int> prefix, long curr, int targetSum) {
        if (root == null) {
            return 0;
        }

        int ret = 0;
        curr += root.val;

        prefix.TryGetValue(curr - targetSum, out ret);
        if (prefix.ContainsKey(curr)) {
            ++prefix[curr];
        } else {
            prefix.Add(curr, 1);
        }
        ret += DFS(root.left, prefix, curr, targetSum);
        ret += DFS(root.right, prefix, curr, targetSum);
        --prefix[curr];

        return ret;
    }
}
```

***

## 2023/7/7

## 416. 分割等和子集

[416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/)
```
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

 

示例 1：

输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
示例 2：

输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。
```

`思路`
动态规划 dp[i][j] 表示0-i范围内 是否能找出和为j的集合

`c# 实现`
```
public class Solution {
    public bool CanPartition(int[] nums) {
        int n = nums.Length;
        int sum = nums.Sum();
        int max = nums.Max();
        int target = sum/2;
        if (n < 2 || (sum & 1) == 1 || max > target)
        {
            return false;
        }
        bool[][] dp = new bool[n][];
        for (int i = 0; i < n; i++)
        {
            dp[i] = new bool[target+1];
            Array.Fill(dp[i],false);
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < target + 1; j++)
            {
                if (j >= nums[i])
                {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
                }else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n - 1][target];
    }
}
```

***

## 2023/7/6

## 406. 根据身高重建队列

[406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/description/)
```
假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

 

示例 1：

输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
示例 2：

输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
```

`思路`
从底开始排

`c# 实现`
```
public class Solution {
    public int[][] ReconstructQueue(int[][] people) {
        Array.Sort(people,(int[] a, int[] b)=>{
            if (a[0] == b[0])
            {
                return b[1] - a[1];
            }else{
                return a[0] - b[0];
            }
        });
        int n = people.Length;
        int[][] ans = new int[n][];
        for (int i = 0; i < n; i++)
        {
            int spaces = people[i][1] + 1;
            for (int j = 0; j < n; j++)
            {
                if (ans[j] == null)
                {
                    spaces--;
                    if (spaces == 0)
                    {
                        ans[j] = people[i];
                        break;
                    }
                }
            }
        }
        return ans;
    }
}
```

***

## 2023/7/5

## 399. 除法求值

[399. 除法求值](https://leetcode.cn/problems/evaluate-division/description/)
```
给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

 

示例 1：

输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
示例 2：

输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
输出：[3.75000,0.40000,5.00000,0.20000]
示例 3：

输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
输出：[0.50000,2.00000,-1.00000,-1.00000]
```

`思路`
转为图

`c# 实现`
```
public class Solution {
    public double[] CalcEquation(IList<IList<string>> equations, double[] values, IList<IList<string>> queries) {
        int nvars = 0;
        Dictionary<string , int> variables = new  Dictionary<string , int>();
        int n = equations.Count;
        for (int i = 0; i < n; i++) {
            if (!variables.ContainsKey(equations[i][0])) {
                variables.Add(equations[i][0],nvars++);
            }
            if (!variables.ContainsKey(equations[i][1])) {
                variables.Add(equations[i][1],nvars++);
            }
        }
        // 对于每个点，存储其直接连接到的所有点及对应的权值
        List<Pair>[] edges = new List<Pair>[nvars];
        for (int i = 0; i < nvars; i++) {
            edges[i] = new List<Pair>();
        }
        for (int i = 0; i < n; i++) {
            int va = variables[equations[i][0]], vb = variables[equations[i][1]];
            edges[va].Add(new Pair(vb, values[i]));
            edges[vb].Add(new Pair(va, 1.0 / values[i]));
        }
        int queriesCount = queries.Count;
        double[] ret = new double[queriesCount];
        for (int i = 0; i < queriesCount; i++)
        {
            IList<String> query = queries[i];
            double result = -1.0;
            if (variables.ContainsKey(query[0]) && variables.ContainsKey(query[1]))
            {
                int ia = variables[query[0]], ib = variables[query[1]];
                if (ia == ib) {
                    result = 1.0;
                } else {
                    Queue<int> points = new Queue<int>();
                    points.Enqueue(ia);
                    double[] ratios = new double[nvars];
                    Array.Fill(ratios, -1.0);
                    ratios[ia] = 1.0;

                    while (points.Count > 0 && ratios[ib] < 0) {
                        int x = points.Dequeue();
                        foreach (Pair pair in edges[x])
                        {
                            int y = pair.index;
                            double val = pair.value;
                            if (ratios[y] < 0) {
                                ratios[y] = ratios[x] * val;
                                points.Enqueue(y);
                            }
                        }
                    }
                    result = ratios[ib];
                }
            }
            ret[i] = result;
        }
        return ret;
    }

 
}

   public class Pair {
        public int index;
        public double value;

        public Pair(int index, double value) {
            this.index = index;
            this.value = value;
        }
    }
```

***

## 2023/7/4

## 394. 字符串解码

[394. 字符串解码](https://leetcode.cn/problems/decode-string/description/)
```
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

 

示例 1：

输入：s = "3[a]2[bc]"
输出："aaabcbc"
示例 2：

输入：s = "3[a2[c]]"
输出："accaccacc"
示例 3：

输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
示例 4：

输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```

`思路`
递归

`c# 实现`
```
public class Solution {
    public string DecodeString(string s) {
        int n = s.Length;
        string temp = String.Empty;
        for (int i = 0; i < n; i++)
        {
            if (char.IsDigit(s[i]))
            {
                int start = i;
                do
                {
                    i++;
                } while (char.IsDigit(s[i]));

                int cnt = int.Parse(s.Substring(start,i-start));
                int balance = 1;
                int j = i;
                do
                {
                    j++;
                    if (s[j] == '[')
                    {
                        balance ++;
                    }else if (s[j] == ']')
                    {
                        balance --;
                    }
                } while (balance != 0);
                string sub = DecodeString(s.Substring(i+1,j - i - 1));
                for (int k = 0; k < cnt; k++)
                {
                    temp += sub;
                }
                i = j;
            }else if(s[i] != '[' && s[i] != ']'){
                temp += s[i];
            }
        }
        return temp;
    }
}
```

***

## 2023/7/3

## 347. 前 K 个高频元素

[347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/)
```
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

 

示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]
```

`思路`
优先队列

`c# 实现`
```
public class Solution {
    public int[] TopKFrequent(int[] nums, int k) {
        int[] ans = new int[k];
        Dictionary<int , int> dict = new Dictionary<int,int>();
        for (int i = 0; i < nums.Length; i++)
        {
            dict.TryAdd(nums[i],0);
            dict[nums[i]]++;
        }
        PriorityQueue<int , int> priQueue = new  PriorityQueue<int, int>();
        foreach( KeyValuePair<int, int> kvp in dict ){
            int val = kvp.Key;
            int count = kvp.Value;
             priQueue.Enqueue(val,count);
            if (priQueue.Count == k + 1)
            {
                priQueue.Dequeue();
            }
        }
        for (int i = 0; i < k; i++)
        {
            ans[i] = priQueue.Dequeue();
        }
        return ans;
    }
}
```

***

## 2023/6/30

## 338. 比特位计数

[338. 比特位计数](https://leetcode.cn/problems/counting-bits/description/)
```
给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。

 

示例 1：

输入：n = 2
输出：[0,1,1]
解释：
0 --> 0
1 --> 1
2 --> 10
示例 2：

输入：n = 5
输出：[0,1,1,2,1,2]
解释：
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public int[] CountBits(int n) {
        int[] bits = new int[n+1];
        for (int i = 1; i <= n; i++)
        {
            bits[i] = bits[i>>1] + (i&1);
        }
        return bits;
    }
}
```

***

## 2023/6/29

## 337. 打家劫舍 III

[打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/description/)
```
小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。

 

示例 1:



输入: root = [3,2,3,null,3,null,1]
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
示例 2:



输入: root = [3,4,5,1,3,null,1]
输出: 9
解释: 小偷一晚能够盗取的最高金额 4 + 5 = 9
```

`思路`
太难了

`c# 实现`
```
public class Solution {
    public int Rob(TreeNode root) {
        int[] ans = DFS(root);
        return Math.Max(ans[0] , ans[1]);
    }

    private int[] DFS(TreeNode root){
        if (root == null)
        {
            return new int[2]{0,0};
        }
        int[] l = DFS(root.left);
        int[] r = DFS(root.right);
        int select = root.val + l[1] + r[1];
        int notSelect = Math.Max(l[0] , l[1]) + Math.Max(r[0] , r[1]);
        return new int[2]{select,notSelect};
    }
}
```

***

## 2023/6/28

## 322. 零钱兑换

[322. 零钱兑换](https://leetcode.cn/problems/coin-change/description/)
```
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

 

示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1
示例 3：

输入：coins = [1], amount = 0
输出：0
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public int CoinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Array.Fill(dp,amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++)
        {   
            for (int j = 0; j < coins.Length; j++)
            {
                if (coins[j] <= i)
                {
                    dp[i] = Math.Min(dp[i - coins[j]] + 1 , dp[i]);
                }
            }
        }
        return dp[amount] > amount ? - 1 : dp[amount];
    }
}
```

***

## 2023/6/27

## 312. 戳气球


[戳气球](https://leetcode.cn/problems/burst-balloons/description/)
```
有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。

 

示例 1：
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
示例 2：

输入：nums = [1,5]
输出：10

```

`思路`

`c# 实现`
```
public class Solution {
    int[][] rec;
    int[] val;
    public int MaxCoins(int[] nums) {
        int n = nums.Length;
        rec = new int[n+2][];
        for (int i = 0; i < n + 2; i++)
        {
            rec[i] = new int[n+2];
            Array.Fill(rec[i],-1);
        }
        val = new int[n + 2];
        val[0] = val[n + 1] = 1;
        for (int i = 0; i < n; i++)
        {
            val[i + 1] = nums[i];
        }
        return Solve(0,n+1);
    }

    private int Solve(int left, int right){
        if (left >= right - 1)
        {
            return 0;
        }
        if (rec[left][right] != -1)
        {
            return rec[left][right];
        }
        for (int i = left + 1; i < right; i++)
        {
            int sum = val[left] * val[i] * val[right];
            sum += Solve(left,i) + Solve(i,right);
            rec[left][right] = Math.Max(sum , rec[left][right]);
        }
        return rec[left][right];
    }
}
```

***

## 2023/6/26

## 309. 最佳买卖股票时机含冷冻期

[309. 最佳买卖股票时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/)
```
给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。​

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 

示例 1:

输入: prices = [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
示例 2:

输入: prices = [1]
输出: 0
```

`思路`
动态规划，主要在于分情况转移，

dp[i][0] 表示当天持有股票的最大累积收益

dp[i][1] 表示当天不持有股票且在冷却的最大累积收益，即当天卖出股票的最大收益

dp[i][2] 表示当天不持有股票且不在冷却的最大累积收益，即前一天卖出或者更早之前就卖出了股票的最大收益

`c# 实现`
```
public class Solution {
    public int MaxProfit(int[] prices) {
        int n = prices.Length;
        int[][] dp = new int[n][];
        for (int i = 0; i < n; i++)
        {
            dp[i] = new int[3];
        }
        dp[0][0] = -prices[0];
        for (int i = 1; i < n; i++)
        {
            dp[i][0] = Math.Max(dp[i - 1][0], dp[i-1][2] - prices[i]);
            dp[i][1] = dp[i - 1][0] + prices[i];
            dp[i][2] = Math.Max(dp[i - 1][1] , dp[i-1][2]);
        }
        return Math.Max(dp[n - 1][1] , dp[n - 1][2]);
    }
}
```

***

## 2023/6/25

## 301. 删除无效的括号

[301. 删除无效的括号](https://leetcode.cn/problems/remove-invalid-parentheses/description/)
```
给你一个由若干括号和字母组成的字符串 s ，删除最小数量的无效括号，使得输入的字符串有效。

返回所有可能的结果。答案可以按 任意顺序 返回。

 

示例 1：

输入：s = "()())()"
输出：["(())()","()()()"]
示例 2：

输入：s = "(a)())()"
输出：["(a())()","(a)()()"]
示例 3：

输入：s = ")("
输出：[""]
```

`思路`
BFS遍历 尝试每种可能

`c# 实现`
```
public class Solution {
    public IList<string> RemoveInvalidParentheses(string s) {
        IList<string> ans = new List<string>();
        HashSet<string> currSet = new HashSet<string>();
        currSet.Add(s);
        while(true){
            foreach (string str in currSet)
            {
                if (IsValid(str))
                {
                    ans.Add(str);
                }
            }
            if (ans.Count > 0)
            {
                return ans;
            }
            HashSet<string> nextSet = new HashSet<string>();
            foreach (string str in currSet)
            {
                for (int i = 0; i < str.Length; i++)
                {
                    if (i > 0 && str[i] == str[i - 1])
                    {
                        continue;
                    }
                    if (str[i] == '(' || str[i] == ')')
                    {
                        nextSet.Add(str.Substring(0,i) + str.Substring(i + 1));
                    }
                }
            }
            currSet = nextSet;
        }
    }

    private bool IsValid(string s){
        int cnt = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(')
            {
                cnt ++;
            }else if (s[i] == ')')
            {
                cnt --;
                if (cnt < 0)
                {
                    return false;
                }
            }
        }
        return cnt == 0;
    }
}
```

***

## 2023/6/21

## 最长递增子序列

[最长递增子序列]()
```
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

 
示例 1：

输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
示例 2：

输入：nums = [0,1,0,3,2,3]
输出：4
示例 3：

输入：nums = [7,7,7,7,7,7,7]
输出：1
```

`思路`
动态规划+二分查找

`c# 实现`
```
public class Solution {
    public int LengthOfLIS(int[] nums) {
        int n = nums.Length;
        int[] dp = new int[n + 1];
        int len = 1;
        dp[len] = nums[0];
        for (int i = 0; i < n; i++)
        {
            if (dp[len] < nums[i])
            {
                dp[++len] = nums[i];
            }else{
                int l = 1, r = len,pos = 0;
                while(l <= r){
                    int mid = (l+r)/2;
                    if (dp[mid] < nums[i])
                    {
                        pos = mid;
                        l = mid + 1;
                    }else{
                        r = mid - 1;
                    }
                }
                dp[pos + 1] = nums[i];
            }
        }
        return len;
    }
}
```

***

## 2023/6/20

## 297. 二叉树的序列化与反序列化

[297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/description/)
```
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

 

示例 1：


输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
示例 2：

输入：root = []
输出：[]
示例 3：

输入：root = [1]
输出：[1]
示例 4：

输入：root = [1,2]
输出：[1,2]

```

`思路`
dfs构造

`c# 实现`
```
public class Codec {

    // Encodes a tree to a single string.
    public string serialize(TreeNode root) {
        return dfsSerialize(root , "");
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(string data) {
        string[] dataArray = data.Split(",");
        LinkedList<string> dataList = new LinkedList<string>(dataArray.ToList());
        return rdeserialize(dataList);
    }

    private string dfsSerialize(TreeNode root , string str){
        if (root == null)
        {
            str += "null,";
        }else{
            str += root.val.ToString() + ",";
            str = dfsSerialize(root.left , str);
            str = dfsSerialize(root.right , str);
        }
        return str;
    }

    public TreeNode rdeserialize(LinkedList<string> dataList) {
        if (dataList.First.Value.Equals("null")) {
            dataList.RemoveFirst();
            return null;
        }
  
        TreeNode root = new TreeNode(int.Parse(dataList.First.Value));
        dataList.RemoveFirst();
        root.left = rdeserialize(dataList);
        root.right = rdeserialize(dataList);
    
        return root;
    }
}
```

***

## 2023/6/19

## 287. 寻找重复数

[287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/description/)
```
给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。

假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。

你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

 

示例 1：

输入：nums = [1,3,4,2,2]
输出：2
示例 2：

输入：nums = [3,1,3,4,2]
输出：3

```

`思路`
二分查找

`c# 实现`
```
public class Solution {
    public int FindDuplicate(int[] nums) {
        int n = nums.Length;
        int l = 0 , r = n - 1;
        int ans = -1;
        while(l <= r){
            int mid = (l+r)/2;
            int cnt = 0;
            for (int i = 0; i < n; i++)
            {
                if (nums[i] <= mid)
                {
                    cnt ++;
                }
            }
            if (cnt <= mid)
            {
                l = mid + 1;
            }else{
                r = mid - 1;
                ans = mid;
            }
        }
        return ans;
    }
}
```

***

## 2023/6/16

## 283. 移动零

[283. 移动零](https://leetcode.cn/problems/move-zeroes/description/)
```
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。

 

示例 1:

输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
示例 2:

输入: nums = [0]
输出: [0]
```

`思路`
双指针

`c# 实现`
```
public class Solution {
    public void MoveZeroes(int[] nums) {
        int n = nums.Length;
        int j = 0;
        for (int i = 0; i < n; i++)
        {
            if (nums[i] != 0)
            {
                swap(nums, i ,j);
                j ++;
            }
        }
    }

    private void swap(int[] nums ,int i ,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

***

## 2023/6/15

## 279. 完全平方数

[279. 完全平方数](https://leetcode.cn/problems/perfect-squares/description/)
```
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

 

示例 1：

输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
示例 2：

输入：n = 13
输出：2
解释：13 = 4 + 9
```

`思路`
动态规划，dp[i] 表示数字i所需的最小个数， 枚举 j * j <= i 的所有j , dp[i] = dp[i- j * j] 的最小一个

`c# 实现`
```
public class Solution {
    public int NumSquares(int n) {
        int[] dp = new int[n+1];
        for (int i = 1; i <= n; i++)
        {
            int min = int.MaxValue;
            for (int j = 1; j * j <= i; j++)
            {
                min = Math.Min(min,dp[i - j*j]);
            }
            dp[i] = min + 1;
        }
        return dp[n];
    }
}
```

***

## 2023/6/14

## 240. 搜索二维矩阵 II

[240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/)
```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
 

示例 1：


输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
示例 2：


输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

`思路`
Z形查找

`c# 实现`
```
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        int n = matrix.Length;
        int m = matrix[0].Length;
        int i = 0, j = m - 1;
        while(i < n && j >= 0){
            if (matrix[i][j] == target)
            {
                return true;
            }
            if (matrix[i][j] > target)
            {
                j--;
            }else{
                i++;
            }
        }
        return false;
    }
}
```

***

## 2023/6/13

## 239. 滑动窗口最大值

[239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)
```
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

 

示例 1：

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
示例 2：

输入：nums = [1], k = 1
输出：[1]
```

`思路`
维护一个list , 保证list的第一个元素是最大的，每次滑动窗口向右，更新这个list，即如果新的元素比第一个元素还大，或者第一个元素已经不在窗口内了，则移除

`c# 实现`
```
public class Solution {
    public int[] MaxSlidingWindow(int[] nums, int k) {
        int n = nums.Length;
        int[] ans = new int[n - k + 1];
        List<int> list = new List<int>();
        for (int i = 0 ; i < k; i++)
        {
            while(list.Count > 0 && nums[i] >= nums[list[list.Count - 1]]){
                list.RemoveAt(list.Count - 1);
            }
            list.Add(i);
        }
        ans[0] = nums[list[0]];
        for (int i = k; i < n; i++)
        {
            while(list.Count > 0 && nums[i] >= nums[list[list.Count - 1]]){
                list.RemoveAt(list.Count - 1);
            }
            list.Add(i);
            if (list[0] <= i - k)
            {
                list.RemoveAt(0);
            }
            ans[i - k + 1] = nums[list[0]];
        }
        return ans;
    }
}
```

***

## 2023/6/12

## 238. 除自身以外数组的乘积

[238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/)
```
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请不要使用除法，且在 O(n) 时间复杂度内完成此题。

 

示例 1:

输入: nums = [1,2,3,4]
输出: [24,12,8,6]
示例 2:

输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
 

提示：

2 <= nums.length <= 105
-30 <= nums[i] <= 30
保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内
 

进阶：你可以在 O(1) 的额外空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
```

`思路`
题目要求不能使用除法

想到一个类似动态规划的思路

dpLeft[i] 记录从0到i的nums的和

dpRight[i] 记录从n-1到i的nums的和

则answer[i] = dpLeft[i] * dpRight[n - i -1]


`c# 实现`
```
public class Solution {
    public int[] ProductExceptSelf(int[] nums) {
        int n = nums.Length;
        int[] left = new int[n+1];
        left[0] = 1;
        int[] right = new int[n+1];
        right[0] = 1;
        for (int i = 1  ; i <= n; i++ )
        {
            left[i] = left[i - 1] * nums[i - 1];
            right[i] = right[i - 1] * nums[n - i];
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++)
        {
            ans[i] = left[i] * right[n - i -1];
        }
        return ans;
    }
}
```

***

## 2023/6/9

## 236. 二叉树的最近公共祖先

[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/)
```
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

 

示例 1：


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
示例 2：


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
示例 3：

输入：root = [1,2], p = 1, q = 2
输出：1
```

`思路`
递归DFS

在root的某子节点有 p 或 q 则返回，否则返回null

存在两种情况

1.公共节点是p q的某一个祖先节点，对应着某个节点的left 和 right递归结果都不为null，此时该节点就是最大祖先节点

2.如果不存在1的情况，公共节点就是某一个节点自己，这种情况直接返回自己

`c# 实现`
```
public class Solution {
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
        {
            return null;
        }
        TreeNode left = LowestCommonAncestor(root.left,p,q);
        TreeNode right = LowestCommonAncestor(root.right,p,q);
        if (left != null && right != null)
        {
            return root;
        }
        if (root == p || root == q)
        {
            return root;
        }
        return left != null ? left : right;
    }
}
```

***

## 2023/6/8

## 234. 回文链表

[234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=featured-list&envId=2cktkvj)
```
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

 

示例 1：


输入：head = [1,2,2,1]
输出：true
示例 2：


输入：head = [1,2]
输出：false
```

`思路`
可以遍历链表，将链表值复制到数组中，再用双指针分别从头 尾遍历，不同则返回false

也可以用快慢指针找到链表的中间，前半部分入栈，后半部分出栈比较

`c# 实现`
```
public class Solution {
    public bool IsPalindrome(ListNode head) {
        bool isHalf = false;
        Stack<int> stack = new Stack<int>();
        ListNode A = head;
        ListNode B = head.next;
        while(A != null){
            if (B != null)
            {
                stack.Push(A.val);
            }
            if (isHalf)
            {
                int curr = stack.Pop();
                if (curr != A.val)
                {
                    return false;
                }
            }
            if (B == null || B.next == null)
            {
                isHalf = true;
                B = null;
            }else{
                B = B.next.next;
            }
            A = A.next;
        }
        return true;
    }
}
```

***

## 2023/6/7

## 226. 翻转二叉树

[226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=featured-list&envId=2cktkvj)
```
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

 

示例 1：



输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
示例 2：



输入：root = [2,1,3]
输出：[2,3,1]
示例 3：

输入：root = []
输出：[]
```

`思路`
BFS 遍历 ， 并交换其 left 和 right

`c# 实现`
```
public class Solution {
    public TreeNode InvertTree(TreeNode root) {
        if(root == null) return root;
        Queue<TreeNode> que = new Queue<TreeNode>();
        que.Enqueue(root);
        while(que.Count > 0){
            for (int i = 0; i < que.Count; i++)
            {
                TreeNode curr = que.Dequeue();
                TreeNode temp = curr.left;
                curr.left = curr.right;
                curr.right = temp;
                if (curr.left != null)
                {
                    que.Enqueue(curr.left);
                }
                if (curr.right != null)
                {
                    que.Enqueue(curr.right);
                }
            }
        }
        return root;
    }
}
```

***

## 2023/6/6

## 221. 最大正方形

[221. 最大正方形](https://leetcode.cn/problems/maximal-square/description/?envType=featured-list&envId=2cktkvj)
```
在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。

 

示例 1：


输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4
示例 2：


输入：matrix = [["0","1"],["1","0"]]
输出：1
示例 3：

输入：matrix = [["0"]]
输出：0

```

`思路`
动态规划
dp[i][j] 表示当前最大的正方形边长

则可得 matrix[i][j] 如果为 ‘1’  dp[i][j] = dp[i-1][j] dp[i][j-1] dp[i-1][j-1] 三个之中最小的一个 加上 1

最后放回最大边长的正方形即可

`c# 实现`
```
public class Solution {
    public int MaximalSquare(char[][] matrix) {
        int n = matrix.Length;
        int m = matrix[0].Length;
        int[][] dp = new int[n+1][];
        for (int i = 0; i <= n; i++)
        {
            dp[i] = new int[m+1];
        }
        int maxWidth = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (matrix[i][j] == '1')
                {
                    int width = Math.Min(dp[i][j] , Math.Min(dp[i+1][j] , dp[i][j+1]));
                    dp[i+1][j+1] = width + 1;
                    maxWidth = Math.Max(maxWidth , dp[i+1][j+1]);
                }
            }
        }
        return maxWidth*maxWidth;
    }
}
```

***

## 2023/6/5

## 215. 数组中的第K个最大元素

[215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=featured-list&envId=2cktkvj)
```
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

 

示例 1:

输入: [3,2,1,5,6,4], k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

`思路`
解法就是实现快排的过程，只是每次排序时只关心k所在的那一半

快排的过程，选一个锚点值 pivot = （0 + nums.Length）/2
将pivot 与 rigth 交换
遍历
遇到比pivot 大的left时， 将nums[right] 赋值为 nums[left]
遇到比pivot 小的rigth时， 将nums[left] 赋值为 nums[right]
重复这个过程直到left == rigth
最后将nums[left] = 一开时的nums[pivot]值

此时nums 被分割为了 [left, pivot - 1] [pivot + 1, right];
一般的快排此时需要递归[left, pivot - 1] 和 [pivot + 1, right] 上述操作
但是此题只关心第 k 大的数，也就是说只需要递归 k所在的 区间即可
最后当pivot == k对应的下标时，就是答案

`c# 实现`
```
 public class Solution {
    int index = 0;
    public int FindKthLargest(int[] nums, int k) {
        index = nums.Length - k;
        return QuickSort(nums,0,nums.Length - 1);
    }

    private int QuickSort(int[] arr, int l , int r){
        if (r <= l)
        {
            return arr[l];
        }
        int pivot = (l + r)/2;
 
        swap(arr , pivot , r);
        pivot = Partition(arr, l , r);
        if (pivot == index)
        {
            return arr[pivot];
        }else{
            return pivot < index ? QuickSort(arr,pivot + 1, r) : QuickSort(arr,l , pivot - 1);
        }
    }

    private int Partition(int[] arr , int l , int r){
        int temp = arr[r];
        while(l != r){
            while(arr[l] <= temp && l < r){
                l++;
            }
            if (l < r)
            {
                arr[r] = arr[l];
                r--;
            }
            while(arr[r] >= temp && r > l){
                r--;
            }
            if (l < r)
            {
                arr[l] = arr[r];
                l++;
            }
        }
        arr[l] = temp;
        return l;
    }

    private void swap(int[] a, int i , int j){
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}
```

***

## 2023/6/2

## 208. 实现 Trie (前缀树)

[208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=featured-list&envId=2cktkvj)
```
Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
 

示例：

输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

`思路`

`c# 实现`
```
public class Trie {
    Dictionary<char,Trie> children;
    bool isEnd;
    public Trie() {
        isEnd = false;
        children = new  Dictionary<char,Trie>();
    }
    
    public void Insert(string word) {
        Trie curr = this;
        for (int i = 0; i < word.Length; i++)
        {
            char c = word[i];
            curr.children.TryAdd(c,new Trie());
            curr = curr.children[c];
        }
        curr.isEnd = true;
    }
    
    public bool Search(string word) {
        Trie curr = this;
        for (int i = 0; i < word.Length; i++)
        {
            char c = word[i];
            if (curr.children.ContainsKey(c))
            {
                curr = curr.children[c];
            }else{
                return false;
            }
        }
        return curr.isEnd;
    }
    
    public bool StartsWith(string prefix) {
        Trie curr = this;
        for (int i = 0; i < prefix.Length; i++)
        {
            char c = prefix[i];
            if (curr.children.ContainsKey(c))
            {
                curr = curr.children[c];
            }else{
                return false;
            }
        }
        return true;
    }
 
}

```

***

## 2023/6/1

## 207. 课程表

[207. 课程表](https://leetcode.cn/problems/course-schedule/description/?envType=featured-list&envId=2cktkvj)
```
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

 

示例 1：

输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
示例 2：

输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```

`思路`
拓扑图算法，BFS遍历

`c# 实现`
```
public class Solution {
    public bool CanFinish(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        Array.Fill(inDegree,0);
        Dictionary<int,List<int>> require = new Dictionary<int,List<int>>();
        for (int i = 0; i < prerequisites.Length; i++)
        {
            inDegree[prerequisites[i][0]]++;
            if (!require.ContainsKey(prerequisites[i][1]))
            {
                require.Add(prerequisites[i][1],new List<int>());
            }
            require[prerequisites[i][1]].Add(prerequisites[i][0]);
        }
        int count = 0;
        Queue<int> que = new Queue<int>();
        for (int i = 0; i < inDegree.Length; i++)
        {
            if (inDegree[i] == 0)
            {
                que.Enqueue(i);
            }
        }
        while(que.Count > 0){
            int curr = que.Dequeue();
            count++;
            if (require.ContainsKey(curr))
            {
                List<int> reqList = require[curr];
                for (int i = 0; i < reqList.Count; i++)
                {
                    inDegree[reqList[i]] --;
                    if (inDegree[reqList[i]] == 0)
                    {
                        que.Enqueue(reqList[i]);
                    }
                }
            }
        }
        return count == numCourses;
    }
}

```

***

## 2023/5/31

## 206. 反转链表

[206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/description/?envType=featured-list&envId=2cktkvj)
```
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
 

示例 1：


输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
示例 2：


输入：head = [1,2]
输出：[2,1]
示例 3：

输入：head = []
输出：[]
```

`思路`
解法1.
递归，类似DFS，先递归到原来的尾节点，再往回赋值

解法2.
遍历链表，用pre ， curr, 记录前一个和当前节点，每次将curr.next = prev;  prev = curr; curr = next;
`c# 实现`
```
1.
public class Solution {
    public ListNode ReverseList(ListNode head) {
        if (head == null || head.next == null)
        {
            return head;
        }
        ListNode temp = ReverseList(head.next);
        head.next.next = head;
        head.next = null;
        return temp;
    }
}

2.
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
}
```

***

## 2023/5/30

## 200. 岛屿数量

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=featured-list&envId=2cktkvj)
```
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

`思路`
递归搜索，搜索过的用另一个二维数组标记

`c# 实现`
```
public class Solution {
    int ans;
    public int NumIslands(char[][] grid) {
        ans = 0;
        int n = grid.Length;
        int m = grid[0].Length;
        bool[][] temp = new  bool[n][];
        for (int i = 0; i < n; i++)
        {
            temp[i] = new bool[m];
            Array.Fill(temp[i],false);
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                 checkIsland(i, j , grid, temp ,false);
            }
        }
        return ans;
    }

    private void checkIsland(int r, int c , char[][] grid, bool[][] temp , bool add){
        int n = grid.Length;
        int m = grid[0].Length;
        if (r<0 || r >= n || c < 0 || c >= m || temp[r][c])
        {
            return;
        }
     
        if (grid[r][c] != '1')
        {
            return;
        }
        temp[r][c] = true;
        if (!add)
        {
            add = true;
            ans ++;
        }
        checkIsland(r - 1, c , grid, temp ,add);
        checkIsland(r + 1, c , grid, temp ,add);
        checkIsland(r, c - 1, grid, temp ,add);
        checkIsland(r, c + 1, grid, temp ,add);
    }
}
```

***

## 2023/5/29

## 198. 打家劫舍

[198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/?envType=featured-list&envId=2cktkvj)
```
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

 

示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 2：

输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

`思路`
动态规划，dp[i] 表示到 i 时 最大收益 ，转移方程则为 dp[i] = Math.Max(dp[i-2] + nums[i-1] , dp[i -1]);

`c# 实现`
```
public class Solution {
    public int Rob(int[] nums) {
        int n = nums.Length;
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = nums[0];
        for (int i = 2; i <= n ; i++)
        {
            dp[i] = Math.Max(dp[i-2] + nums[i-1] , dp[i -1]);
        }
        return dp[n];
    }
}
```

***

## 2023/5/26

## 169. 多数元素

[169. 多数元素](https://leetcode.cn/problems/majority-element/description/)
```
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

示例 1：

输入：nums = [3,2,3]
输出：3
示例 2：

输入：nums = [2,2,1,1,1,2,2]
输出：2
```

`思路`
排序
Boyer-Moore 投票算法

`c# 实现`
```
我的解法
public class Solution {
    public int MajorityElement(int[] nums) {
        Array.Sort(nums);
        int major = nums[0];
        int max = 1;
        int left = 1;
        for (int i = 1; i < nums.Length; i++)
        {
            if (nums[i] != nums[i - 1] || i == nums.Length - 1)
            {
                if (i - left + 1 > max)
                {
                    max = i - left + 1;
                    major = nums[i - 1];
                }
                left = i;
            }
             
        }
        return major;
    }
}

官方
public class Solution {
    public int MajorityElement(int[] nums) {
        Array.Sort(nums);
        return nums[nums.Length / 2];
    }
}

public class Solution {
    public int MajorityElement(int[] nums) {
        int cnt = 0;
        int candiate = nums[0];
        for(int i = 0; i < nums.Length; i++){
            if(cnt == 0){
                candiate = nums[i];
            }
            cnt += (nums[i] == candiate) ? 1 : -1;
        }
        return candiate;
    }
}
```

***

## 2023/5/25

## 160. 相交链表

[160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/)
```
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：



题目数据 保证 整个链式结构中不存在环。

注意，函数返回结果后，链表必须 保持其原始结构 。

自定义评测：

评测系统 的输入如下（你设计的程序 不适用 此输入）：

intersectVal - 相交的起始节点的值。如果不存在相交节点，这一值为 0
listA - 第一个链表
listB - 第二个链表
skipA - 在 listA 中（从头节点开始）跳到交叉节点的节点数
skipB - 在 listB 中（从头节点开始）跳到交叉节点的节点数
评测系统将根据这些输入创建链式数据结构，并将两个头节点 headA 和 headB 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 视作正确答案 。

 

示例 1：



输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
— 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。
 

示例 2：



输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
示例 3：



输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。
```

`思路`
解法一：
哈希表记录
解法二：
遍历
具体参考[这篇解答](https://leetcode.cn/problems/intersection-of-two-linked-lists/solutions/10774/tu-jie-xiang-jiao-lian-biao-by-user7208t/)

`c# 实现`
```
一：
public class Solution {
    public ListNode GetIntersectionNode(ListNode headA, ListNode headB) {
        HashSet<ListNode> set = new HashSet<ListNode>();
        while(headA != null || headB != null){
            if (headA == headB)
            {
                return headA;
            }
            if (set.Contains(headA))
            {
                return headA;
            }
            if (set.Contains(headB))
            {
                return headB;
            }
            if (headA != null)
            {
                set.Add(headA);
                headA = headA.next;
            }
            if (headB != null)
            {
                set.Add(headB);
                headB = headB.next;
            }

        }
        return null;
    }
}

二：
public class Solution {
    public ListNode GetIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null)
        {
            return null;
        }
        ListNode pa = headA , pb = headB;
        while(pa != pb){
            pa = pa == null ? headB : pa.next;
            pb = pb == null ? headA : pb.next;
        }
        return pa;
    }
}
```

***

## 2023/5/24

## 155. 最小栈

[155. 最小栈](https://leetcode.cn/problems/min-stack/description/)
```
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
 

示例 1:

输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

`思路`
辅助栈。每次push辅助栈都push栈顶和val比较小的那个，getmin时就能保证拿到辅助栈的栈顶是最小值

`c# 实现`
```
public class MinStack {
    Stack<int> stack;
    Stack<int> minStack;
    public MinStack() {
        stack = new Stack<int>();
        minStack = new Stack<int>();
    }
    
    public void Push(int val) {
        stack.Push(val);
        if(minStack.Count > 0){
             minStack.Push(Math.Min(minStack.Peek() , val));
        }else{
            minStack.Push(val);
        }
       
    }
    
    public void Pop() {
        stack.Pop();
        minStack.Pop();
    }
    
    public int Top() {
        return stack.Peek();
    }
    
    public int GetMin() {
        return minStack.Peek();
    }
}
```

***

## 2023/5/23

## 152. 乘积最大子数组

[152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/)
```
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 32-位 整数。

子数组 是数组的连续子序列。

 

示例 1:

输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
示例 2:

输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

`思路`
动态规划，dp[i] 来表示以第 i 个元素结尾的乘积最大子数组的乘积
因为有负数，需要额外一个dpMin来记录 以第 i 个元素结尾的乘积最小子数组的乘积
每次在dpMax dpMin nums[i] 中比较最大值

`c# 实现`
```
public class Solution {
    public int MaxProduct(int[] nums) {
        int n = nums.Length;
        int[] dpMax = new int[n];
        int[] dpMin = new int[n];
        dpMax[0] = nums[0];
        dpMin[0] = nums[0];
        for (int i = 1; i < n; i++)
        {
            dpMax[i] = Math.Max(dpMax[i - 1] * nums[i] , Math.Max(dpMin[i - 1] * nums[i] , nums[i]));
            dpMin[i] = Math.Min(dpMin[i - 1] * nums[i] , Math.Min(dpMax[i - 1] * nums[i] , nums[i]));
        }
        int ans = int.MinValue;
        for (int i = 0; i < n; i++)
        {
            ans = Math.Max(ans,dpMax[i]);
        }
        return ans;
    }
}
```

***

## 2023/5/22

## 148. 排序链表

[148. 排序链表](https://leetcode.cn/problems/sort-list/description/)
```
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

 

示例 1：


输入：head = [4,2,1,3]
输出：[1,2,3,4]
示例 2：


输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
示例 3：

输入：head = []
输出：[]
```

`思路`
解法一：
直接遍历记录值，排序后重新赋值

解法二：
分段合并链表

`c# 实现`
```
public class Solution {
    public ListNode SortList(ListNode head) {
        if (head == null) {
            return head;
        }
        int length = 0;
        ListNode node = head;
        while (node != null) {
            length++;
            node = node.next;
        }
        ListNode dummyHead = new ListNode(0, head);
        for (int subLength = 1; subLength < length; subLength *= 2)
        {
            ListNode prev = dummyHead, curr = dummyHead.next;
            while (curr != null) {
                ListNode head1 = curr;
                for (int i = 1; i < subLength && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode head2 = curr.next;
                curr.next = null;
                curr = head2;
                for (int i = 1; i < subLength && curr != null && curr.next != null; i++)
                {
                    curr = curr.next;
                }
                ListNode next = null;
                if (curr != null) {
                    next = curr.next;
                    curr.next = null;
                }
                ListNode merged = Merge(head1,head2);
                prev.next = merged;
                while (prev.next != null) {
                    prev = prev.next;
                }
                curr = next;
            }
        }
        return dummyHead.next;
    }

    private ListNode Merge(ListNode head1 , ListNode head2){
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while(temp1 != null && temp2 != null){
            if (temp1.val <= temp2.val)
            {
                temp.next = temp1;
                temp1 = temp1.next;
            }else{
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null)
        {
            temp.next = temp1;
        }else if (temp2 != null)
        {
            temp.next = temp2;
        }
        return dummyHead.next;
    }
}
```

***

## 2023/5/19

## 146. LRU 缓存

[146. LRU 缓存](https://leetcode.cn/problems/lru-cache/description/)
```
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

 

示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4

```

`思路`
需要使用哈希表+双向链表来实现

实现上使用假的head和tail来实现将很方便

`c# 实现`
```
public class LRUCache {

    public class MyLinkedList{
        public int val;
        public int key;
        public MyLinkedList next;
        public MyLinkedList prev;
        public MyLinkedList(int key = 0 ,int val = 0){
            this.key = key;
            this.val = val;
        }
    }

    int capacity;
    Dictionary<int, MyLinkedList> dict;
    MyLinkedList head;
    MyLinkedList tail;
    public LRUCache(int capacity) {
        this.capacity = capacity;
        dict = new Dictionary<int, MyLinkedList>();
        // 使用伪头部和伪尾部节点
        head = new MyLinkedList();
        tail = new MyLinkedList();
        head.next = tail;
        tail.prev = head;
    }
    
    public int Get(int key) {
        if (!dict.ContainsKey(key))
        {
            return -1;
        }
        MyLinkedList node = dict[key];
        moveToHead(node);
        return node.val;
    }
    
    public void Put(int key, int value) {
        if (dict.ContainsKey(key))
        {
            dict[key].val = value;
            moveToHead(dict[key]);
            return;
        }
        if (dict.Count >= capacity)
        {
            MyLinkedList remove = removeTail();
            dict.Remove(remove.key);
        }
        MyLinkedList newNode = new MyLinkedList(key,value);
        addToHead(newNode);
        dict.Add(key,newNode);
    }

   private void addToHead(MyLinkedList node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }


    private void removeNode(MyLinkedList node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(MyLinkedList node) {
        removeNode(node);
        addToHead(node);
    }

    private MyLinkedList removeTail() {
        MyLinkedList res = tail.prev;
        removeNode(res);
        return res;
    }
}
```

***

## 2023/5/18

## 142. 环形链表 II

[142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/)
```
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

 

示例 1：



输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
示例 2：



输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
示例 3：



输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

`思路`
一：
哈希表记录

二：
快慢指针，两个指针A B。 A每次移动一个，B每次移动两个，如果存在环，A一定会追上B ， 即 A == B ， 否则A,B最终位null
根据题意，任意时刻，fast\textit{fast}fast 指针走过的距离都为 slow\textit{slow}slow 指针的 222 倍。因此，我们有

a+(n+1)b+nc=2(a+b)  ⟹  a=c+(n−1)(b+c) a+(n+1)b+nc=2(a+b) \implies a=c+(n-1)(b+c)
a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)
有了 a=c+(n−1)(b+c)a=c+(n-1)(b+c)a=c+(n−1)(b+c) 的等量关系，我们会发现：从相遇点到入环点的距离加上 n−1n-1n−1 圈的环长，恰好等于从链表头部到入环点的距离。

因此，当发现 slow\textit{slow}slow 与 fast\textit{fast}fast 相遇时，我们再额外使用一个指针 ptr\textit{ptr}ptr。起始，它指向链表头部；随后，它和 slow\textit{slow}slow 每次向后移动一个位置。最终，它们会在入环点相遇。




`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode DetectCycle(ListNode head) {
        HashSet<ListNode> set = new HashSet<ListNode>();
        while(head != null){
            if (set.Contains(head))
            {
                return head;
            }
            set.Add(head);
            head = head.next;
        }
        return null;
    }
}
```

***

## 2023/5/18

## 141. 环形链表

[141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/description/)
```
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

 

示例 1：



输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
示例 2：



输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
示例 3：



输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

`思路`
一：
哈希表记录

二：
快慢指针，两个指针A B。 A每次移动一个，B每次移动两个，如果存在环，A一定会追上B ， 即 A == B ， 否则A,B最终位null

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
1:

public class Solution {
    public bool HasCycle(ListNode head) {
        HashSet<ListNode> set = new HashSet<ListNode>();
        while(head != null){
            if (set.Contains(head))
            {
                return true;
            }
            set.Add(head);
            head = head.next;
        }
        return false;
    }
}

2：

public class Solution {
    public bool HasCycle(ListNode head) {
        if (head == null || head.next == null)
        {
            return false;
        }
        ListNode A = head;
        ListNode B = head.next;
        while(A != B){
            if (B == null || B.next == null)
            {
                return false;
            }
            A = A.next;
            B = B.next.next;
        }
        return true;
    }
}
```

***

## 2023/5/18

## 139. 单词拆分

[139. 单词拆分](https://leetcode.cn/problems/word-break/description/)
```
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

 

示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

`思路`
只用字典树会被 "aaaa" "aa" s = "aaaaaaa" 这种用例卡住。
这道题还是得动态规划，用 dp[i] 表示在i位置时，能否用wordDict里的单词拼接，dp[i] = dp[j] && wordDict.Contains(s.Substring(j , i - j))
其中j是 0 到 i 的某一下标， s.Substring(j , i - j)是其对应的字符串片段

`c# 实现`
```
public class Solution {

    public bool WordBreak(string s, IList<string> wordDict) {
        HashSet<string> set = new HashSet<string>(wordDict);
        int n = s.Length;
        bool[] dp = new bool[n + 1];
        dp[0] = true;
        for (int i = 1; i <= n; i++)
        {
            for (int j = i; j >= 0; j--)
            {
                if (dp[j] && set.Contains(s.Substring(j , i - j)))
                {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

***

## 2023/5/17

## 136. 只出现一次的数字

[136. 只出现一次的数字](https://leetcode.cn/problems/single-number/description/?favorite=2cktkvj)
```
给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

 

示例 1 ：

输入：nums = [2,2,1]
输出：1
示例 2 ：

输入：nums = [4,1,2,1,2]
输出：4
示例 3 ：

输入：nums = [1]
输出：1
```

`思路`
解法一  排序后遍历

解法二 位运算

交换律：a ^ b ^ c <=> a ^ c ^ b

任何数于0异或为任何数 0 ^ n => n

相同的数异或为0: n ^ n => 0

var a = [2,3,2,4,4]

2 ^ 3 ^ 2 ^ 4 ^ 4等价于 2 ^ 2 ^ 4 ^ 4 ^ 3 => 0 ^ 0 ^3 => 3

`c# 实现`
```
public class Solution {
    public int SingleNumber(int[] nums) {
        Array.Sort(nums);
        int n = nums.Length;
        for (int i = 0; i < n; i+=2)
        {
            if (i == n - 1 || nums[i] != nums[i + 1])
            {
                return nums[i];
            }
        }
        return 0;
    }
}


public class Solution {
    public int SingleNumber(int[] nums) {
        int n = nums.Length;
        int ans = nums[0];
        for (int i = 1; i < n; i++)
        {
            ans ^= nums[i];
        }
        return ans;
    }
}
```

***

## 2023/5/16

## 128. 最长连续序列

[128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
```
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

 

示例 1：

输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
示例 2：

输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```

`思路`
哈希表记录

`c# 实现`
```
public class Solution {
    public int LongestConsecutive(int[] nums) {
        ISet<int> longset = new HashSet<int>();
        for(int i = 0; i < nums.Length; i ++){
            longset.Add(nums[i]);
        }
        int longest = 0;
        foreach (int num in longset)
        {
            if (!longset.Contains(num - 1))
            {
                int cur = 1;
                int index = 1;
                while(longset.Contains(num + index)){
                    cur ++;
                    index ++;
                }
                longest = Math.Max(longest,cur);
            }
        }
        return longest;
    }
}
```

***

## 2023/5/15

## 124. 二叉树中的最大路径和

[124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/?favorite=2cktkvj)
```
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

 

示例 1：


输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
示例 2：


输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

`思路`
递归

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    int ans;
    public int MaxPathSum(TreeNode root) {
        ans = int.MinValue;
        Sum(root);
        return ans;
    }

    private int Sum(TreeNode root){
        if (root == null)
        {
            return 0;
        }
        int left = Math.Max(Sum(root.left) , 0);
        int right = Math.Max(Sum(root.right) , 0);
        int sum = left + right + root.val;
        ans = Math.Max(sum,ans);
        return Math.Max(left,right) + root.val;
    }
}
```

***

## 2023/5/12

## 121. 买卖股票的最佳时机

[121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?favorite=2cktkvj)
```
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

 

示例 1：

输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
示例 2：

输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
```

`思路`
用min记录当前最小的一个

`c# 实现`
```
public class Solution {
    public int MaxProfit(int[] prices) {
        int n = prices.Length;
        int min = prices[0];
        int profit = 0;
        for (int i = 1; i < n; i++)
        {
            min = Math.Min(min , prices[i]);
            profit = Math.Max(profit, prices[i] - min);
        }
        return profit;
    }
}
```

***

## 2023/5/11

## 114. 二叉树展开为链表

[114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?favorite=2cktkvj)
```
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
 

示例 1：


输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
示例 2：

输入：root = []
输出：[]
示例 3：

输入：root = [0]
输出：[0]
```

`思路`
先序遍历

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public void Flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode ans = null;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.Push(root);
        while(stack.Count > 0){
            TreeNode cur = stack.Pop();
            if (ans != null)
            {
                ans.left = null;
                ans.right = cur;
            }
            if (cur.right != null)
            {
                stack.Push(cur.right);
            }
            if (cur.left != null)
            {
                stack.Push(cur.left);
            }
            ans = cur;
        }
    }

}
```

***

## 2023/5/11

## 105. 从前序与中序遍历序列构造二叉树

[105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?favorite=2cktkvj)
```
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

 

示例 1:


输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]
示例 2:

输入: preorder = [-1], inorder = [-1]
输出: [-1]
```

`思路`
递归

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    private Dictionary<int,int> indexDict;
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
        int n = preorder.Length;
        indexDict = new Dictionary<int,int>();
        for (int i = 0; i < n; i++)
        {
            indexDict.Add(inorder[i],i);
        }
        return Build(preorder,inorder,0,n-1,0);
    }

    private TreeNode Build(int[] preorder, int[] inorder , int preLeft , int preRight, int inLeft ){
        if (preLeft > preRight)
        {
            return null;
        }
        int val = preorder[preLeft];
        TreeNode root = new TreeNode(val);
        int rootInInOrder = indexDict[val];
        int leftLen = rootInInOrder - inLeft;

        root.left =  Build(preorder, inorder , preLeft + 1 , preLeft + leftLen, inLeft);
        root.right = Build(preorder, inorder , preLeft + leftLen + 1 , preRight , rootInInOrder + 1 );
        return root;
    }
}
```

***

## 2023/5/9

## 104. 二叉树的最大深度

[104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?favorite=2cktkvj)
```
给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。
```

`思路`
DFS

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int MaxDepth(TreeNode root) {
        return DFS(root,0);
    }

    private int DFS(TreeNode root , int depth){
        if (root == null)
        {
            return depth;
        }
        depth ++;
        return Math.Max(DFS(root.left,depth) , DFS(root.right,depth));
    }
}
```

***

## 2023/5/8

## 102. 二叉树的层序遍历

[102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?favorite=2cktkvj)
```
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

 

示例 1：


输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
示例 2：

输入：root = [1]
输出：[[1]]
示例 3：

输入：root = []
输出：[]
```

`思路`
队列

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public IList<IList<int>> LevelOrder(TreeNode root) {
        IList<IList<int>> ans = new List<IList<int>>();
        if (root == null)
        {
            return ans;
        }
        Queue<TreeNode> que = new  Queue<TreeNode>();
        que.Enqueue(root);
        while(que.Count > 0){
            IList<int> temp = new List<int>();
            int count = que.Count;
            for (int i = 0; i < count; i++)
            {
                TreeNode nd = que.Dequeue();
                temp.Add(nd.val);
                if (nd.left != null)
                {
                    que.Enqueue(nd.left);
                }
                if (nd.right != null)
                {
                    que.Enqueue(nd.right);
                }
            }
            ans.Add(temp);
        }
        return ans;
    }
}
```

***

## 2023/5/6

## 101. 对称二叉树

[101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/description/?favorite=2cktkvj)
```
给你一个二叉树的根节点 root ， 检查它是否轴对称。

 

示例 1：


输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：


输入：root = [1,2,2,null,3,null,3]
输出：false
```

`思路`
递归比较

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool IsSymmetric(TreeNode root) {
        return Check(root,root);
    }

    private bool Check(TreeNode l,TreeNode r){
        if (l == null && r == null)
        {
            return true;
        }
        if (l == null || r == null)
        {
            return false;
        }
        return l.val == r.val && Check(l.left,r.right) && Check(l.right,r.left);
    }
}
```

***

## 2023/5/5

## 98. 验证二叉搜索树

[98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/?favorite=2cktkvj)
```
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

有效 二叉搜索树定义如下：

节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
 

示例 1：


输入：root = [2,1,3]
输出：true
示例 2：


输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
```

`思路`
中序遍历二叉树

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool IsValidBST(TreeNode root) {
        long inorder = long.MinValue;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while(root != null || stack.Count > 0){
            while(root != null){
                stack.Push(root);
                root = root.left;
            }
            root = stack.Pop();
            if (root.val <= inorder)
            {
                return false;
            }
            inorder = root.val;
            root = root.right;
        }
        return true;
    }
}

解法2

public class Solution {
    long inorder;
    public bool IsValidBST(TreeNode root) {
        inorder = long.MinValue;
        return IsValid(root);
    }

    private bool IsValid(TreeNode root){
        if (root == null)
        {
            return true;
        }
        bool left = IsValid(root.left);
        if (!left)
        {
            return false;
        }
        if (root.val <= inorder)
        {
            return false;
        }
        inorder = root.val;
        return IsValid(root.right);
    }
}
```

***

## 2023/5/4

## 96. 不同的二叉搜索树

[96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/description/?favorite=2cktkvj)
```
给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

 

示例 1：


输入：n = 3
输出：5
示例 2：

输入：n = 1
输出：1
```

`思路`
递归每次找节点左右的组合
还可以直接动态规划实现 dp[i] 记为 i个节点时，有多少个组合

`c# 实现 递归`
```
public class Solution {
    public int NumTrees(int n) {
        return GetNum(n);
    }

    private int GetNum(int n){
        if (n < 3)
        {
            return Math.Max(n,1);
        }
        int res = 0;
        for (int l = 0 , r = n - 1; l < n; l++ , r--)
        {
            res += GetNum(l) * GetNum(r);
        }
        return res;
    }
}
```

`c++ 动态规划`
```
class Solution {
public:
    int numTrees(int n) {
        vector<int> G(n + 1, 0);
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }
};

```

***

## 2023/4/28

## 94. 二叉树的中序遍历

[94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?favorite=2cktkvj)
```
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

 

示例 1：


输入：root = [1,null,2,3]
输出：[1,3,2]
示例 2：

输入：root = []
输出：[]
示例 3：

输入：root = [1]
输出：[1]
```

`思路`
中序遍历

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public IList<int> InorderTraversal(TreeNode root) {
        IList<int> ans = new List<int>();
        DFS(root,ans);
        return ans;
    }

    private void DFS(TreeNode root , IList<int> ans){
        if (root == null)
        {
            return;
        }
        DFS(root.left,ans);
        ans.Add(root.val);
        DFS(root.right,ans);
    }
}
```

***

## 2023/4/27

## 85. 最大矩形

[85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/description/?favorite=2cktkvj)
```
给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

 

示例 1：


输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
示例 2：

输入：matrix = []
输出：0
示例 3：

输入：matrix = [["0"]]
输出：0
示例 4：

输入：matrix = [["1"]]
输出：1
示例 5：

输入：matrix = [["0","0"]]
输出：0
```

`思路`
这道题需要结合[84. 柱状图中最大的矩形]来做
其实就是将矩阵转换成了很多列柱状图最大面积的问题

`c# 实现`
```
public class Solution {
    public int MaximalRectangle(char[][] matrix) {
        int m = matrix.Length;
        if (m == 0)
        {
            return 0;
        }
        int n = matrix[0].Length;
        int[][] left = new int[m][];
        for (int i = 0; i < m; i++)
        {
            left[i] = new int[n];
            for (int j = 0; j < n; j++)
            {
                if (matrix[i][j] == '1')
                {
                     left[i][j] = (j == 0 ? 0 : left[i][j - 1] ) + 1;
                }
            }
        }
        int ans = 0;
        for (int j = 0; j < n; j++)
        {
            int[] up = new int[m];
            int[] down = new int[m];
            Stack<int> stack = new Stack<int>();
            for (int i = 0; i < m; i++)
            {
                while(stack.Count > 0 && left[stack.Peek()][j] >= left[i][j]){
                    stack.Pop();
                }
                up[i] = stack.Count > 0 ? stack.Peek() : -1;
                stack.Push(i);
            }
            stack = new Stack<int>();
            for (int i = m - 1; i >= 0; i--)
            {
                while(stack.Count > 0 && left[stack.Peek()][j] >= left[i][j]){
                    stack.Pop();
                }
                down[i] = stack.Count > 0 ? stack.Peek() : m;
                stack.Push(i);
            }
            for (int i = 0; i < m; i++)
            {
                ans = Math.Max(ans,(down[i] - up[i] - 1) * left[i][j]);
            }
        }
        return ans;
    }
}
```

***

## 2023/4/26

## 84. 柱状图中最大的矩形

[84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?favorite=2cktkvj)
```
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

示例 1:



输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
示例 2：



输入： heights = [2,4]
输出： 4
```

`思路`
单调栈，用left[i] right[i] 分别记录下标i时，最多能到的左右边界

`c# 实现`
```
public class Solution {
    public int LargestRectangleArea(int[] heights) {
        int ans = 0;
        int n = heights.Length;
        int[] left = new int[n];
        int[] right = new int[n];
        Stack<int> stack = new Stack<int>();
        for (int i = 0; i < n; i++)
        {
            while(stack.Count > 0 && heights[stack.Peek()] >= heights[i]){
                stack.Pop();
            }
            left[i] = stack.Count > 0 ? stack.Peek() : -1;
            stack.Push(i);
        }
        stack.Clear();
        for (int i = n - 1; i >= 0; i--)
        {
            while(stack.Count > 0 && heights[stack.Peek()] >= heights[i]){
                stack.Pop();
            }
            right[i] = stack.Count > 0 ? stack.Peek() : n;
            stack.Push(i);
        }
        for (int i = 0; i < n; i++)
        {
            ans = Math.Max(ans,(right[i] - left[i] - 1)*heights[i]);
        }
        return ans;
    }
}
```

***

## 2023/4/25

## 79. 单词搜索

[79. 单词搜索](https://leetcode.cn/problems/word-search/description/?favorite=2cktkvj)
```
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

 

示例 1：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true
示例 3：


输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
输出：false
```

`思路`
递归回溯

`c# 实现`
```
public class Solution {
    public bool Exist(char[][] board, string word) {
        int m = board.Length;
        int n = board[0].Length;
        bool[][] visited = new bool[m][];
        for (int i = 0; i < m; i++)
        {
            visited[i] = new bool[n];
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                
                if (DFS(board,i,j,word,0 , visited))
                {
                    return true;
                }
            }
        }
        return false;
    }

    private bool DFS(char[][] board , int i  , int j ,string word , int index ,bool[][] visited){
        if (i < 0 || j < 0 || i >= board.Length || j >= board[0].Length)
        {
            return false;
        }
        if (visited[i][j])
        {
            return false;
        }
        if (board[i][j] != word[index])
        {
            return false;
        }else{
            if (index == word.Length - 1)
            {
                return true;
            }else{
                visited[i][j] = true;
                bool l = DFS(board, i - 1, j , word, index + 1,visited);
                bool r = DFS(board, i + 1, j , word, index + 1,visited);
                bool u = DFS(board, i , j + 1, word, index + 1,visited);
                bool d = DFS(board, i , j - 1, word, index + 1,visited);
                visited[i][j] = false;
                return l || r || u || d;
            }
        }
    }
}
```

***

## 2023/4/24

## 78. 子集

[78. 子集](https://leetcode.cn/problems/subsets/description/?favorite=2cktkvj)
```
给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

 

示例 1：

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：

输入：nums = [0]
输出：[[],[0]]
```

`思路`
倒序遍历添加

`c# 实现`
```
public class Solution {
    public IList<IList<int>> Subsets(int[] nums) {
        IList<IList<int>> ans = new List<IList<int>>();
        ans.Add(new List<int>());
        int n = nums.Length;
        for (int i = n - 1; i >= 0; i--)
        {
            int n2 = ans.Count;
            for (int j = 0; j < n2; j++)
            {
                List<int> temp = new List<int>(ans[j]);
                temp.Add(nums[i]);
                ans.Add(temp);
            }
        }
        return ans;
    }
}
```

***

## 2023/4/23

## 76. 最小覆盖子串

[76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/?favorite=2cktkvj)
```
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

 

注意：

对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
 

示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
示例 2：

输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
示例 3:

输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
```

`思路`
滑动窗口

`c# 实现`
```
public class Solution {
    Dictionary<char,int> ori = new Dictionary<char,int>();
    Dictionary<char,int> cnt = new Dictionary<char,int>();
    public string MinWindow(string s, string t) {
        for (int i = 0; i < t.Length; i++)
        {
            ori.TryAdd(t[i],0);
            ori[t[i]]++;
        }
        int n = s.Length;
        int l = 0, r = 0;
        string ans = "";
        int minLen = int.MaxValue;
        while(r < n){
            if (ori.ContainsKey(s[r]))
            {
                cnt.TryAdd(s[r],0);
                cnt[s[r]] ++;
            }
            while(Check() && l <= r){
                if (r - l + 1 < minLen)
                {
                    minLen = r - l + 1;
                    ans = s.Substring(l,minLen);
                }
                if (ori.ContainsKey(s[l]))
                {
                    cnt[s[l]] --;
                }
                l++;
            }
            r++;
        }
        return ans;
    }

    private bool Check(){
        foreach (KeyValuePair<char, int> kvp in ori)
        {
            if (!cnt.ContainsKey(kvp.Key) || cnt[kvp.Key] < kvp.Value)
            {
                return false;
            }
        }
        return true;
    }
}
```

***

## 2023/4/21

## 75. 颜色分类

[75. 颜色分类](https://leetcode.cn/problems/sort-colors/description/?favorite=2cktkvj)
```
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

 

示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
示例 2：

输入：nums = [2,0,1]
输出：[0,1,2]
```

`思路`
实现一个快速排序

`c# 实现`
```
 public class Solution {
    public void SortColors(int[] nums) {
        QuickSort(nums,0,nums.Length - 1);
    }

    private void QuickSort(int[] nums, int left , int right){
        if (left >= right)
        {
            return;
        }
        int baseNum = nums[left];
        int i = left;
        int j = right;
        while(i < j){
            while(i < j && nums[j] >= baseNum){
                j --;
            }
            if (i < j)
            {
                nums[i++] = nums[j];
            }
            while(i < j && nums[i] <= baseNum){
                i ++;
            }
            nums[j] = nums[i];
        }
        nums[i] = baseNum;
        QuickSort(nums,left,i - 1);
        QuickSort(nums,i + 1,right);
    }
}
```

***

## 2023/4/20

## 72. 编辑距离

[72. 编辑距离](https://leetcode.cn/problems/edit-distance/description/?favorite=2cktkvj)
```
给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
 

示例 1：

输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：

输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

`思路`
动态规划，dp[i][j] 表示word1在第i个字符 和 word2在第j个字符时最少需要操作的次数

`c# 实现`
```
public class Solution {
    public int MinDistance(string word1, string word2) {
        int m = word1.Length ,n = word2.Length;
        if(m*n == 0){
            return Math.Max(m,n);
        }
        int[][] dp = new int[m + 1][];
        for (int i = 0; i < m + 1; i++)
        {
            dp[i] = new int[n + 1];
            dp[i][0] = i;
        }
        for (int i = 0; i < n + 1; i++)
        {
            dp[0][i] = i;
        }
        for (int i = 1; i < m + 1; i++)
        {
            for (int j = 1; j < n + 1; j++)
            {
                if (word1[i - 1] != word2[j - 1])
                {
                    dp[i][j] = 1 + Math.Min(Math.Min(dp[i - 1][j] , dp[i][j - 1]) , dp[i - 1][j - 1]);
                }else{
                    dp[i][j] = 1 + Math.Min(Math.Min(dp[i - 1][j] , dp[i][j - 1]) , dp[i - 1][j - 1] - 1);
                }
            }
        }
        return dp[m][n];
    }
}
```

***

## 2023/4/19

## 70. 爬楼梯

[70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?favorite=2cktkvj)
```
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

 

示例 1：

输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
示例 2：

输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
```

`思路`
动态规划，dp[i] 表示i位置时有多少种走法，则 dp[i] = dp[i-1] + dp[i-2]

`c# 实现`
```
public class Solution {
    public int ClimbStairs(int n) {
        int[] temp = new int[n + 1];
        temp[0] = 1;
        temp[1] = 1;
        for (int i = 2; i <= n; i++)
        {
            temp[i] = temp[i - 1] + temp[i - 2];
        }
        return temp[n];
    }
}
```

***

## 2023/4/18

## 64. 最小路径和

[64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/description/?favorite=2cktkvj)
```
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

 

示例 1：


输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
示例 2：

输入：grid = [[1,2,3],[4,5,6]]
输出：12
```

`思路`
动态规划，dp[i][j] 表示矩阵 i j 位置时， 最小的和

`c# 实现`
```
public class Solution {
    public int MinPathSum(int[][] grid) {
        int m =grid.Length;
        int n = grid[0].Length;
        int[][] dp = new int[m][];
        for (int i = 0; i < m; i++)
        {
            dp[i] = new int[n];
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == 0 && j == 0)
                {
                    dp[i][j] = grid[i][j];
                    continue;
                }
                int l = j > 0 ? dp[i][j-1] : int.MaxValue;
                int u = i > 0 ? dp[i-1][j] : int.MaxValue;
                dp[i][j] = Math.Min(l,u) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }
}
```

***

## 2023/4/17

## 62. 不同路径

[62. 不同路径](https://leetcode.cn/problems/unique-paths/description/?favorite=2cktkvj)
```
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

 

示例 1：


输入：m = 3, n = 7
输出：28
示例 2：

输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
示例 3：

输入：m = 7, n = 3
输出：28
示例 4：

输入：m = 3, n = 3
输出：6
```

`思路`
动态规划，dp[i][j]表示到下标 i j 位置，总共有多少种路线 dp[i][j] = dp[i-1][j] + dp[i][j-1]

`c# 实现`
```
public class Solution {
    public int UniquePaths(int m, int n) {
        int[][] dp = new int[m][];
        for (int i = 0; i < m; i++)
        {
            dp[i] = new int[n];
        }
        // dp[0][0] = 1;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if(i == 0 || j == 0){
                    dp[i][j] = 1;
                }else{
                    int l = dp[i][j - 1];
                    int up = dp[i - 1][j];
                    dp[i][j] = l + up;
                }
               
            }
        }
        return dp[m - 1][n - 1];
    }
}
```

***

## 2023/4/14

## 56. 合并区间

[56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/?favorite=2cktkvj?favorite=2cktkvj)
```
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

 

示例 1：

输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2：

输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```

`思路`
排序后遍历比较区间

`c# 实现`
```
public class Solution {
    public int[][] Merge(int[][] intervals) {
        Array.Sort(intervals,(int[] a, int[] b)=>{
            return a[0] - b[0];
        });
        int n = intervals.Length;
        List<int[]> ans = new List<int[]>();
        for (int i = 0; i < n; i++)
        {
            int l = intervals[i][0] ,  r = intervals[i][1];
            if (ans.Count == 0 || ans[ans.Count - 1][1] < l)
            {
                ans.Add(new int[2]{l,r});
            }else{
               ans[ans.Count - 1][1] = Math.Max(r,ans[ans.Count - 1][1]);
            }
        }
        return ans.ToArray();
    }
}
```

***

## 2023/4/13

## 55. 跳跃游戏

[55. 跳跃游戏](https://leetcode.cn/problems/jump-game/description/?favorite=2cktkvj?favorite=2cktkvj)
```
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

 

示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
示例 2：

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```

`思路`
贪心算法，遍历nums，记录当前能到达的最大下标max，遍历时要判断当前下标是否超过了max，如果max超过了最后一位下标，则可以到达

`c# 实现`
```
public class Solution {
    public bool CanJump(int[] nums) {
        int n = nums.Length;
        int max = nums[0];
        for (int i = 0; i < n; i++)
        {
            if (i <= max)
            {
                max = Math.Max(max, i + nums[i]);
                if (max >= n - 1)
                {
                    return true;
                }
            }
        }
        return false;
    }
}
```

***

## 2023/4/12

## 53. 最大子数组和

[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?favorite=2cktkvj)
```
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

 

示例 1：

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
示例 2：

输入：nums = [1]
输出：1
示例 3：

输入：nums = [5,4,-1,7,8]
输出：23
```

`思路`
dp[i] 来表示以当前i结尾时，最大的子串大小，所以可得 ： dp[i + 1] = (dp[i] > 0 ? dp[i] : 0) + nums[i + 1]

`c# 实现`
```
public class Solution {
    public int MaxSubArray(int[] nums) {
        int n = nums.Length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        int res = dp[0];
        for (int i = 1; i < n; i++)
        {
            if (dp[i - 1] > 0)
            {
                dp[i] = Math.Max(nums[i] , dp[i - 1] + nums[i]);
            }else{
                dp[i] = nums[i];
            }
            res = Math.Max(dp[i] , res);
        }
        return res;
    }
}

//简化版
public class Solution {
    public int MaxSubArray(int[] nums) {
        int n = nums.Length;
        int res = nums[0];
        int pre = 0;
        for (int i = 0; i < n; i++)
        {
            pre = Math.Max(nums[i], pre + nums[i]);
            res = Math.Max(res,pre);
        }
        return res;
    }
}
```

***

## 2023/4/11

## 49. 字母异位词分组

[49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/?favorite=2cktkvj)
```
你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

 

示例 1:

输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
示例 2:

输入: strs = [""]
输出: [[""]]
示例 3:

输入: strs = ["a"]
输出: [["a"]]

```

`思路`
将排序后的单词哈希表记录分组

`c# 实现`
```
public class Solution {
    public IList<IList<string>> GroupAnagrams(string[] strs) {
        Dictionary<string,IList<string>> dict = new Dictionary<string,IList<string>>();
        IList<IList<string>> ans = new List<IList<string>>();
        for (int i = 0; i < strs.Length; i++)
        {
            char[] temp = strs[i].ToCharArray();
            Array.Sort(temp);
            string tempStr = new string(temp);
            dict.TryAdd(tempStr,new List<string>());
            dict[tempStr].Add(strs[i]);
        }

        foreach (KeyValuePair<string, IList<string>> kvp in dict)
        {
            ans.Add(new List<string>(kvp.Value));
        }
        return ans;
    }
}
```

***

## 2023/4/10

## 48. 旋转图像

[48. 旋转图像](https://leetcode.cn/problems/rotate-image/description/?favorite=2cktkvj)
```
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

 

示例 1：


输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
示例 2：


输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

`思路`
主要在于不能用新的矩阵记录，所以只有在每次变换时记录第一个matrix[i][j]

`c# 实现`
```
public class Solution {
    public void Rotate(int[][] matrix) {
        int n = matrix.Length;
        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < (n + 1) / 2 ; j++)
            {
                int temp =  matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[n - (n - j - 1) - 1][n - i - 1];
                matrix[n - (n - j - 1) - 1][n - i - 1] = temp;
            }
        }
    }
}
```

***

## 2023/4/6

## 42. 接雨水

[42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/description/?favorite=2cktkvj)
```
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

示例 1：



输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
示例 2：

输入：height = [4,2,0,3,2,5]
输出：9
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public int Trap(int[] height) {
       int n = height.Length;
       if (n < 2)
       {
            return 0;
       }
       int res = 0;
       int[] leftMax = new int[n];
       leftMax[0] = height[0];
       for (int i = 1; i < n; i++)
       {
            leftMax[i] = Math.Max(leftMax[i - 1] , height[i]);
       }
       int[] rigthMax = new int[n];
       rigthMax[n - 1] = height[n - 1];
       for (int i = n - 2; i >= 0; i--)
       {
            rigthMax[i] = Math.Max(rigthMax[i + 1] , height[i]);
       }
       for (int i = 0; i < n; i++)
       {
            res += Math.Min(leftMax[i] , rigthMax[i]) - height[i];
       }
       return res;
    }
}
```

***

## 2023/4/4

## 39. 组合总和

[39. 组合总和](https://leetcode.cn/problems/combination-sum/description/?favorite=2cktkvj)
```
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

 

示例 1：

输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
示例 2：

输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
示例 3：

输入: candidates = [2], target = 1
输出: []

```

`思路`
搜索回溯
需要考虑每种情况的题目应用回溯算法，直到满足条件

`c# 实现`
```
public class Solution {
    public IList<IList<int>> CombinationSum(int[] candidates, int target) {
        IList<IList<int>> ans = new List<IList<int>>();
        List<int> combine = new List<int>();
        Dfs(candidates,target,ans,combine,0);
        return ans;
    }

    private void Dfs(int[] candidates , int target , IList<IList<int>> ans , List<int> combine , int idx){
        if (idx == candidates.Length)
        {
            return;
        }
        if (target == 0)
        {
            ans.Add(new List<int>(combine));
            return;
        }
        Dfs(candidates,target,ans,combine,idx + 1);
        if (target - candidates[idx] >= 0)
        {
            combine.Add(candidates[idx]);
            Dfs(candidates,target - candidates[idx],ans,combine,idx);
            combine.RemoveAt(combine.Count - 1);
        }
    }
}
```

***

## 2023/4/3

## 34. 在排序数组中查找元素的第一个和最后一个位置

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?favorite=2cktkvj)
```
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

 

示例 1：

输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
示例 2：

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
示例 3：

输入：nums = [], target = 0
输出：[-1,-1]

```

`思路`
要求O(log n),所以基本只能是二分查找。
先二分查找到target的某个位置，再往前和往后延伸寻找相同的位置。

`c# 实现`
```
public class Solution {
    public int[] SearchRange(int[] nums, int target) {
        int[] res = new int[2]{-1,-1};
        int n = nums.Length;
        if (n > 0)
        {
            int l = 0;
            int r = n - 1;
            bool find = false;
            while(l <= r){
                int mid = (l + r + 1) / 2;
                if (target == nums[mid])
                {
                    
                    l = mid;
                    find = true;
                    break;
                }
                else if (target > nums[mid])
                {
                    l = mid + 1;
                }else{
                    r = mid - 1;
                }
            }
            if (find)
            {
                
                int index = l;
                while(index - 1 >= 0 && index - 1 < n && nums[index-1] == nums[l]){
                    index --;
                }
                res[0] = index;
                index = l;
                while(index + 1 >= 0 && index + 1 < n && nums[index + 1] == nums[l]){
                    index ++;
                }
                res[1] = index;
            }
        
        }
        return res;
    }
}
```

***

## 2023/3/31

## 33. 搜索旋转排序数组

[33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?favorite=2cktkvj)
```
整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

 

示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
示例 2：

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
示例 3：

输入：nums = [1], target = 0
输出：-1
```

`思路`
先判断target在哪个递增区，再二分查找

`c# 实现`
```
public class Solution {
    public int Search(int[] nums, int target) {
        int n = nums.Length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0 , r = n - 1;
        while(l <= r){
            int mid = (l+r)/2;
            if (nums[mid] == target)
            {
                return mid;
            }
            if (nums[mid] >= nums[0])
            {
                if (target >= nums[0] && target < nums[mid])
                {
                    r = mid - 1;
                }else{
                    l = mid + 1;
                }
            }else{
                if (target > nums[mid] && target <= nums[n - 1])
                {
                    l = mid + 1;
                }else{
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

***

## 2023/3/30

## 32. 最长有效括号

[32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/description/?favorite=2cktkvj)
```
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

 

示例 1：

输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
示例 2：

输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
示例 3：

输入：s = ""
输出：0
```

`思路`
动态规划，dp[i]表示当前最长有效长度

`c# 实现`
```
public class Solution {
    public int LongestValidParentheses(string s) {
        int n = s.Length;
        int[] dp = new int[n+1];
        Array.Fill(dp,0);
        for (int i = 1; i < n; i++)
        {
            if (s[i] == ')' && s[i - 1] == '(')
            {
                dp[i] = (i >= 2 ? dp[i-2] : 0) + 2;
            }else if (s[i] == ')' && s[i - 1] == ')')
            {
                if(i - dp[i-1]  > 0 && s[i - dp[i - 1] - 1] == '('){
                     dp[i] = dp[i-1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
               
            }
        }
        return dp.Max();
    }
}
```

***

## 2023/3/29

## 31. 下一个排列

[31. 下一个排列](https://leetcode.cn/problems/next-permutation/description/?favorite=2cktkvj)
```
整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
给你一个整数数组 nums ，找出 nums 的下一个排列。

必须 原地 修改，只允许使用额外常数空间。

 

示例 1：

输入：nums = [1,2,3]
输出：[1,3,2]
示例 2：

输入：nums = [3,2,1]
输出：[1,2,3]
示例 3：

输入：nums = [1,1,5]
输出：[1,5,1]

```

`思路`
从右往左找到第一个比右边还小的位置（因为交换位数越低越好，此位置即为要交换的位置 i），再次从右往左找到第一个比位置i 大的 位置 j, 交换i j ,最后再从i + 1开始反转后面的数组

`c# 实现`
```
public class Solution {
    public void NextPermutation(int[] nums) {
        int n = nums.Length;
        int i = n - 2;
        while(i >= 0 && nums[i] >= nums[i + 1]){
            i --;
        }
        if (i >= 0)
        {
            int j = n - 1;
            while(j >= 0 && nums[i] >= nums[j]){
                j --;
            }
            swap(nums,i , j);
        }
        reverse(i + 1,nums);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int start , int[] nums){
        int left = start;
        int right = nums.Length - 1;
        while(left < right){
            swap(nums, left, right);
            left ++;
            right --;
        }
    }

}
```

***

## 2023/3/28

## 25. K 个一组翻转链表

[25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/)
```
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

 

示例 1：


输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
示例 2：



输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
```

`思路`
数组记录要反转的节点

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode ReverseKGroup(ListNode head, int k) {
        if (k < 2)
        {
            return head;
        }
        ListNode[] list = new ListNode[k];
        ListNode preHead = new ListNode(-1);
        ListNode pre = preHead;
        ListNode cur = head;
        int cnt = 0;
        while(cur != null){
            list[cnt] = cur;
            cur = cur.next;
            cnt++;
            if (cnt == k)
            {
                for (int i = k - 1; i >=0; i--)
                {
                    if (i == k - 1)
                    {
                        pre.next = list[i];
                        list[i].next = list[i - 1];
                    }else if(i == 0){
                        list[i].next = cur;
                        pre = list[i];
                    }else{
                        list[i].next = list[i - 1];
                    }
                }
                cnt = 0;
            }
        }
        return preHead.next;
    }
}
```

***

## 2023/3/27

## 24. 两两交换链表中的节点

[24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/)
```
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

 

示例 1：


输入：head = [1,2,3,4]
输出：[2,1,4,3]
示例 2：

输入：head = []
输出：[]
示例 3：

输入：head = [1]
输出：[1]
```

`思路`
用prehead 做前置节点

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode SwapPairs(ListNode head) {
        ListNode preHead = new ListNode(-1);
        ListNode pre = preHead;
        ListNode cur = head;
        while(cur != null && cur.next != null){
            ListNode temp = cur.next;
            cur.next = cur.next.next;
            temp.next = cur;
            pre.next = temp;
            pre = cur;
            cur = cur.next;
        }
        return preHead.next == null ? head : preHead.next;
    }
}
```

***

## 2023/3/24

## 23. 合并 K 个升序链表

[23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/description/?favorite=2cktkvj)
```
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
示例 2：

输入：lists = []
输出：[]
示例 3：

输入：lists = [[]]
输出：[]
```

`思路`

解法一：
直接便利链表数组，每次找出最小的一个元素接到答案链表上，直到所有链表都为空

解法二：
使用优先队列记录每个链表当前位置的节点，每次链接出列的节点并添加下一个非空节点

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
 解法一 ： 顺序遍历
 public class Solution {
    public ListNode MergeKLists(ListNode[] lists) {
        if(lists.Length == 0){
            return null;
        }
        ListNode preHead = new ListNode(-1);
        ListNode curNd = preHead;
        while(true){
            int nullCnt = 0;
            int minIndex = 0;
            int min = int.MaxValue;
            for (int i = 0; i < lists.Length; i++)
            {
                if (lists[i] == null)
                {
                    nullCnt ++;
                    continue;
                }
                if (min > lists[i].val)
                {
                    min = lists[i].val;
                    minIndex = i;
                }
            }
            if (nullCnt == lists.Length )
            {
                break;
            }
            curNd.next = lists[minIndex];
            curNd = curNd.next;
            lists[minIndex] = lists[minIndex].next;
           
        }
        return preHead.next;
    }
}
 
解法二 ： 优先队列
public class Solution {
    public ListNode MergeKLists(ListNode[] lists) {
        ListNode preHead = new ListNode(-1);
        ListNode curNd = preHead;
        PriorityQueue<ListNode,int> que = new PriorityQueue<ListNode,int>();
        for (int i = 0; i < lists.Length; i++)
        {
            if (lists[i] != null)
            {
                que.Enqueue(lists[i],lists[i].val);
            }
        }
        while(que.Count > 0){
            ListNode temp = que.Dequeue();
            curNd.next = temp;
            curNd = curNd.next;
            if(temp.next != null){
                que.Enqueue(temp.next,temp.next.val);
            }
            
        }
        return preHead.next;
    }
}
```

***

## 2023/3/23

## 22. 括号生成

[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/description/?favorite=2cktkvj)
```
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

 

示例 1：

输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
示例 2：

输入：n = 1
输出：["()"]
```

`思路`
递归每个个数的括号情形并记录

`c# 实现`
```
public class Solution {
    IList<string>[] cache = new IList<string>[100];
    public IList<string> GenerateParenthesis(int n) {
        return GenParent(n);
    }

    private IList<string> GenParent(int n){
        if (cache[n] != null)
        {
            return cache[n];
        }
        IList<string> ans = new List<string>();
        if (n == 0)
        {
            ans.Add("");
        }
        
        for (int i = 0; i < n; i++)
        {
            foreach (string left in GenParent(i))
            {
                foreach (string right in GenParent(n - i - 1))
                {
                    ans.Add("("+left+")"+right);
                }
            }
        }
        cache[n] = ans;
        return ans;
    }
}
```

***

## 2023/3/22

## 21. 合并两个有序链表

[21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/?favorite=2cktkvj)
```
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 

示例 1：


输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
示例 2：

输入：l1 = [], l2 = []
输出：[]
示例 3：

输入：l1 = [], l2 = [0]
输出：[0]
```

`思路`
再头部加一个前头节点，遍历比较两个节点并依次指向小的那个，直到某一个为空

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode MergeTwoLists(ListNode list1, ListNode list2) {
     
        ListNode preHead = new ListNode(-1);
        ListNode cur = preHead;
        while(list1 != null && list2 != null){
            if (list1.val > list2.val)
            {
                cur.next = list2;
                list2 = list2.next;
            }else{
                cur.next = list1;
                list1 = list1.next;
            }
            cur = cur.next;
        }
        cur.next = list1 == null ? list2 : list1;
        return preHead.next;
    }
}
```

***

## 2023/3/21

## 20. 有效的括号

[20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/description/?favorite=2cktkvj)
```
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
 

示例 1：

输入：s = "()"
输出：true
示例 2：

输入：s = "()[]{}"
输出：true
示例 3：

输入：s = "(]"
输出：false
```

`思路`
左括号入栈，右括号出栈匹配

`c# 实现`
```
public class Solution {
    public bool IsValid(string s) {
        Stack<char> stack = new Stack<char>();
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(' || s[i] == '{' || s[i] == '[')
            {
                stack.Push(s[i]);
            }else{
                if (stack.Count == 0)
                {
                    return false;
                }
                char c = stack.Pop();
                bool isMatch = false;
                switch (c)
                {
                    case '(':{
                        isMatch = s[i] == ')';
                        break;
                    }
                    case '[':{
                        isMatch = s[i] == ']';
                        break;
                    }
                    case '{':{
                        isMatch = s[i] == '}';
                        break;
                    }
                }
                if (!isMatch)
                {
                    return false;
                }
            }
        }
        return stack.Count == 0;
    }
}
```

***

## 2023/3/20

## 19. 删除链表的倒数第 N 个结点

[19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?favorite=2cktkvj)
```
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

 

示例 1：


输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：

输入：head = [1], n = 1
输出：[]
示例 3：

输入：head = [1,2], n = 1
输出：[1]
```

`思路`
哈希表记录节点，也可以入栈删除

`c# 实现`
```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode RemoveNthFromEnd(ListNode head, int n) {
        Dictionary<int , ListNode> dict = new Dictionary<int , ListNode>();
        int index = 0;
        ListNode curNd = head;
        while(curNd != null){
            dict.TryAdd(index ++,curNd);
            curNd = curNd.next;
        }
        int tarIndex = index - n;
        ListNode target = dict[tarIndex];
        if (target == head)
        {
            head = head.next;
        }else{
            dict[tarIndex - 1].next = dict[tarIndex].next;
        }
        return head;
    }
}
```

***

## 2023/3/17

## 17. 电话号码的字母组合

[17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?favorite=2cktkvj)
```
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。



 

示例 1：

输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：

输入：digits = ""
输出：[]
示例 3：

输入：digits = "2"
输出：["a","b","c"]
```

`思路`
回溯

`c# 实现`
```
public class Solution {
    public IList<string> LetterCombinations(string digits) {
        IList<string> ans = new List<string>();
        if (digits.Length == 0) {
            return ans;
        }
        Dictionary<char , string> dict = new Dictionary<char , string>();
        dict.Add('2', "abc");
        dict.Add('3', "def");
        dict.Add('4', "ghi");
        dict.Add('5', "jkl");
        dict.Add('6', "mno");
        dict.Add('7', "pqrs");
        dict.Add('8', "tuv");
        dict.Add('9', "wxyz");

        BackTrack(ans,dict ,0,digits,"");
        return  ans;
    }

    private void BackTrack(IList<string> ans ,  Dictionary<char , string> dict, int index, string digits , string comb){
        if (index == digits.Length)
        {
            ans.Add(comb);
            return;
        }
        string str = dict[digits[index]];
        for (int i = 0; i < str.Length; i++)
        {
            BackTrack(ans,dict,index + 1,digits,comb + str[i]);
        }
    }
}
```

***

## 2023/3/16

## 15. 三数之和

[15. 三数之和](https://leetcode.cn/problems/3sum/description/?favorite=2cktkvj)
```
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

 

 

示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
示例 2：

输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
示例 3：

输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
```

`思路`
排序+双指针
主要在于如何去除重复答案
1.可以将目标list排序后转成字符串记录在哈希表，缺点是比较麻烦而且空间
2.数组已经排过序，所以遍历时只要保证下一次数和当前数不同即可排除重复答案

`c# 实现`
```
public class Solution {
    public IList<IList<int>> ThreeSum(int[] nums) {
        IList<IList<int>> ans = new List<IList<int>>();
        Array.Sort(nums);
        int n = nums.Length;
        for (int i = 0; i < n - 2; i++)
        {
            if (nums[i] > 0)
            {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.Length - 1;
            while(left < right){
                int sum = nums[left] + nums[right];
                if (sum == -nums[i])
                {
                    List<int> list = new List<int>(){nums[i],nums[left],nums[right]};
                    ans.Add(list);
                    while(left < right && nums[left] == nums[left + 1]){
                        left ++;
                    }
                    while(left < right && nums[right] == nums[right - 1]){
                        right --;
                    }
                    left ++;
                    right --;
                }else{
                    if (sum + nums[i] > 0)
                    {
                        right --;
                    }else{
                        left ++;
                    }
                }
            }
        }
        return ans;
    }
}
```

***

## 2023/3/15

## 11. 盛最多水的容器

[11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/?favorite=2cktkvj)
```
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

 

示例 1：



输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
示例 2：

输入：height = [1,1]
输出：1

```

`思路`
双指针，开始指向两头，每次移动较小的那一个

`c# 实现`
```
public class Solution {
    public int MaxArea(int[] height) {
        int n = height.Length;
        int res = 0;
        int left = 0 , right = n - 1;
        while(left < right){
            int area = Math.Min(height[left] , height[right]) * (right - left);
            res = Math.Max(area,res);
            if (height[left] > height[right])
            {
                right --;
            }else{
                left ++;
            }
        }
        return res;
    }
}
```

***

## 2023/3/14

## 10. 正则表达式匹配

[10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/description/?favorite=2cktkvj)
```
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

 
示例 1：

输入：s = "aa", p = "a"
输出：false
解释："a" 无法匹配 "aa" 整个字符串。
示例 2:

输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
示例 3：

输入：s = "ab", p = ".*"
输出：true
解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

`思路`
动态规划

`c# 实现`
```
public class Solution {
    public bool IsMatch(string s, string p) {
        int sLen = s.Length;
        int pLen = p.Length;
        bool[][] dp = new bool[sLen+1][];
        for (int i = 0; i < sLen+1; i++)
        {
            dp[i] = new bool[pLen+1];
            Array.Fill(dp[i],false);
            // dp[i].Fill(false);
        }
        dp[0][0] = true;
        for (int j = 1; j < pLen + 1; j++) {
            if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];
        }
        for (int i = 1; i < sLen + 1; i++)
        {
            for (int j = 1; j < pLen + 1; j++)
            {
                if (s[i - 1] == p[j - 1] || p[j - 1] == '.')
                {
                    dp[i][j] = dp[i-1][j-1];
                }else if(p[j - 1] == '*'){
                    if (s[i - 1] == p[j - 2] || p[j - 2] == '.') {
                        dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
                    } else {
                        dp[i][j] = dp[i][j - 2];
                    }
                }
            }
        }
        return dp[sLen][pLen];
    }
}
```

***

## 2023/3/13

## 5. 最长回文子串

[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/?favorite=2cktkvj)
```
给你一个字符串 s，找到 s 中最长的回文子串。

如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

 

示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"
```

`思路`
动态规划，记dp[i][j]为第i到j的子字符串是否为回文字符串

`c# 实现`
```
public class Solution {
    public string LongestPalindrome(string s) {
        int len = s.Length;
        if (len <= 1)
        {
            return s;
        }
        int maxLen = 1;
        int begin = 0;
        bool[][] dp = new bool[len][];
        for (int i = 0; i < len; i++)
        {
            dp[i] = new bool[len];
            dp[i][i] = true;
        }
        for (int L = 2; L <= len; L++)
        {
            for (int i = 0 ; i < len; i++)
            {
                int j = i + L - 1;
                if (j >= len) {
                    break;
                }
                if (s[i] == s[j])
                {
                    if (j - i < 3)
                    {
                        dp[i][j] = true;
                    }else{
                        dp[i][j] = dp[i+1][j-1];
                    }
                }else{
                    dp[i][j] = false;
                }
                if (dp[i][j] == true && j - i + 1 > maxLen)
                {
                    maxLen = j - i +1;
                    begin = i;
                }
            }
        }
        return s.Substring(begin,maxLen);
    }
}
```

***

## 2023/3/10

## 4. 寻找两个正序数组的中位数

[4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?favorite=2cktkvj)
```
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 。

 

示例 1：

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
示例 2：

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

`思路`
要想到O(log(m+n)),想到的是二分查找，递归二分查找

`c# 实现`
```
public class Solution {
    public double FindMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.Length;
        int n = nums2.Length;
        int left = (m + n + 1)/2;
        int right = (m + n + 2)/2;
       return ( (double)(GetMedian(nums1,0,m - 1,nums2,0,n - 1,left) + GetMedian(nums1,0,m - 1,nums2,0,n - 1,right)) * 0.5);
    }

    private double GetMedian(int[] nums1, int start1 , int end1 , int[] nums2, int start2, int end2 , int k){
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        if (len1 > len2)
        {
            return GetMedian(nums2,start2,end2,nums1,start1,end1,k);    
        }
        if (len1 == 0)
        {
            return nums2[start2 + k -1];
        }
        if (k == 1)
        {
            return Math.Min(nums1[start1],nums2[start2]) ;
        }
        int i = start1 + Math.Min(len1 ,k/2) - 1;
        int j = start2 + Math.Min(len2 ,k/2) - 1;
        if (nums1[i] > nums2[j])
        {
            
            return GetMedian(nums1,start1,end1,nums2,j + 1,end2,k - (j - start2 + 1));
        }else{
            return GetMedian(nums1,i+1,end1,nums2,start2,end2,k - (i - start1 +1));
        }
    }
}
```

***

## 2023/3/9

## 3. 无重复字符的最长子串

[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?favorite=2cktkvj)
```
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

`思路`
双指针滑动窗口，遍历字符串，left right之间为当前子串

`c# 实现`
```
public class Solution {
    public int LengthOfLongestSubstring(string s) {
        Dictionary<char,int> dict = new Dictionary<char,int>();
        int max = 0;
        int n = s.Length;
        for (int left = 0 ,  right = 0; right < n; right++)
        {
            if (dict.ContainsKey(s[right]))
            {
                left =Math.Max(dict[s[right]] + 1 , left);
                dict[s[right]] = right;
            }else{
                dict.Add(s[right],right);
            }
            max = Math.Max(max,right - left + 1);
        }
        return max;
    }
}
```

***

## 2023/3/8

## 剑指 Offer 47. 礼物的最大价值

[剑指 Offer 47. 礼物的最大价值](https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/description/)
```
在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

 

示例 1:

输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

`思路`
动态规划，记 f(i,j) 表示从棋盘的左上角走到位置 (i,j)，最多可以拿到的礼物的价值 , 遍历grid, i , j位置的最大值 = Math.Max(f(i-1,j) , f(i,j-1)) 最后返回 f[m - 1][n - 1];

`c# 实现`
```
public class Solution {
    public int MaxValue(int[][] grid) {
        int m = grid.Length, n = grid[0].Length;
        int[][] f = new int[m][];
        for (int i = 0; i < m; ++i) {
            f[i] = new int[n];
            for (int j = 0; j < n; ++j) {
                if (i > 0) {
                    f[i][j] = Math.Max(f[i][j], f[i - 1][j]);
                }
                if (j > 0) {
                    f[i][j] = Math.Max(f[i][j], f[i][j - 1]);
                }
                f[i][j] += grid[i][j];
            }
        }
        return f[m - 1][n - 1];
    }
}


### 我自己的实现， 虽然没想到用动态规划，但实际思路是差不多的，只是空间复杂度和时间复杂度略高，勉强没有超过时间限制
public class Solution {
    private Dictionary<string,int> dict = new Dictionary<string,int>();
    public int MaxValue(int[][] grid) {
        return GetMax(0,0,grid);
    }

    private int GetMax(int row , int col, int[][] grid){
        if (dict.ContainsKey(row+"-"+col))
        {
           return dict[row+"-"+col];
        }
        int right = 0;
        int down = 0;
        if (row < grid.Length - 1)
        {
            down = GetMax(row + 1,col , grid);
        }
        if (col < grid[0].Length - 1)
        {
            right = GetMax(row ,col + 1 , grid);
        }
        int max = grid[row][col] + Math.Max(right,down);
        dict.TryAdd(row+"-"+col,max);
        return max;
    }
}
```

***

## 2023/3/6

## 1653. 使字符串平衡的最少删除次数

[1653. 使字符串平衡的最少删除次数](https://leetcode.cn/problems/minimum-deletions-to-make-string-balanced/description/)
```
给你一个字符串 s ，它仅包含字符 'a' 和 'b'​​​​ 。

你可以删除 s 中任意数目的字符，使得 s 平衡 。当不存在下标对 (i,j) 满足 i < j ，且 s[i] = 'b' 的同时 s[j]= 'a' ，此时认为 s 是 平衡 的。

请你返回使 s 平衡 的 最少 删除次数。

 

示例 1：

输入：s = "aababbab"
输出：2
解释：你可以选择以下任意一种方案：
下标从 0 开始，删除第 2 和第 6 个字符（"aababbab" -> "aaabbb"），
下标从 0 开始，删除第 3 和第 6 个字符（"aababbab" -> "aabbbb"）。
示例 2：

输入：s = "bbaaaaabb"
输出：2
解释：唯一的最优解是删除最前面两个字符。
```

`思路`
遍历数组，当前下标左边的所有b加上当前下标右边的所有a数量的和即为当前下标所需的操作数，遍历完成后返回最小的一个操作数

`c# 实现`
```
public class Solution {
    public int MinimumDeletions(string s) {
        int righta = 0;
        int leftb = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == 'a')
            {
                righta ++;
            }
        }
        int res = righta;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == 'a')
            {
                righta --;
            }else{
                leftb ++;
            }
            res = Math.Min(res, righta + leftb);
        }
        return res;
    }
}
```

***

## 2023/3/3

## 1487. 保证文件名唯一

[1487. 保证文件名唯一](https://leetcode.cn/problems/making-file-names-unique/description/)
```
给你一个长度为 n 的字符串数组 names 。你将会在文件系统中创建 n 个文件夹：在第 i 分钟，新建名为 names[i] 的文件夹。

由于两个文件 不能 共享相同的文件名，因此如果新建文件夹使用的文件名已经被占用，系统会以 (k) 的形式为新文件夹的文件名添加后缀，其中 k 是能保证文件名唯一的 最小正整数 。

返回长度为 n 的字符串数组，其中 ans[i] 是创建第 i 个文件夹时系统分配给该文件夹的实际名称。

 

示例 1：

输入：names = ["pes","fifa","gta","pes(2019)"]
输出：["pes","fifa","gta","pes(2019)"]
解释：文件系统将会这样创建文件名：
"pes" --> 之前未分配，仍为 "pes"
"fifa" --> 之前未分配，仍为 "fifa"
"gta" --> 之前未分配，仍为 "gta"
"pes(2019)" --> 之前未分配，仍为 "pes(2019)"
示例 2：

输入：names = ["gta","gta(1)","gta","avalon"]
输出：["gta","gta(1)","gta(2)","avalon"]
解释：文件系统将会这样创建文件名：
"gta" --> 之前未分配，仍为 "gta"
"gta(1)" --> 之前未分配，仍为 "gta(1)"
"gta" --> 文件名被占用，系统为该名称添加后缀 (k)，由于 "gta(1)" 也被占用，所以 k = 2 。实际创建的文件名为 "gta(2)" 。
"avalon" --> 之前未分配，仍为 "avalon"
示例 3：

输入：names = ["onepiece","onepiece(1)","onepiece(2)","onepiece(3)","onepiece"]
输出：["onepiece","onepiece(1)","onepiece(2)","onepiece(3)","onepiece(4)"]
解释：当创建最后一个文件夹时，最小的正有效 k 为 4 ，文件名变为 "onepiece(4)"。
示例 4：

输入：names = ["wano","wano","wano","wano"]
输出：["wano","wano(1)","wano(2)","wano(3)"]
解释：每次创建文件夹 "wano" 时，只需增加后缀中 k 的值即可。
示例 5：

输入：names = ["kaido","kaido(1)","kaido","kaido(1)"]
输出：["kaido","kaido(1)","kaido(2)","kaido(1)(1)"]
解释：注意，如果含后缀文件名被占用，那么系统也会按规则在名称后添加新的后缀 (k) 。
```

`思路`
哈希表

`c# 实现`
```
public class Solution {
    public string[] GetFolderNames(string[] names) {
        IDictionary<string, int> index = new Dictionary<string, int>();
        int n = names.Length;
        string[] res = new string[n];
        for (int i = 0; i < n; i++)
        {
            string name = names[i];
            if (!index.ContainsKey(name))
            {
                res[i] = name;
                index.Add(name,1);
            }else{
                int k = index[name];
                while(index.ContainsKey(AddSuffix(name,k))){
                    k++;
                }
                res[i] = AddSuffix(name,k);
                index[name] = k + 1;
                index.Add(AddSuffix(name,k),1);
            }
        }
        return res;
    }
     public string AddSuffix(string name, int k) {
        return name + "(" + k + ")";
    }
}
```

***

## 2023/3/2

## 面试题 05.02. 二进制数转字符串

[面试题 05.02. 二进制数转字符串](https://leetcode.cn/problems/bianry-number-to-string-lcci/description/)
```
二进制数转字符串。给定一个介于0和1之间的实数（如0.72），类型为double，打印它的二进制表达式。如果该数字无法精确地用32位以内的二进制表示，则打印“ERROR”。

示例1:

 输入：0.625
 输出："0.101"
示例2:

 输入：0.1
 输出："ERROR"
 提示：0.1无法被二进制准确表示
 

提示：

32位包括输出中的 "0." 这两位。
题目保证输入用例的小数位数最多只有 6 位
```

`思路`
主要在于小数如何用二进制表示

`c# 实现`
```
public class Solution {
    public string PrintBin(double num) {
        StringBuilder sb = new StringBuilder("0.");
        while(num != 0 && sb.Length <= 32){
            num *= 2;
            int digit = (int) num;
            sb.Append(digit);
            num -= digit;
        }
        return sb.Length <= 32 ? sb.ToString() : "ERROR";
    }
}
```

***

## 2023/3/1

## 2373. 矩阵中的局部最大值

[2373. 矩阵中的局部最大值](https://leetcode.cn/problems/largest-local-values-in-a-matrix/description/)
```
给你一个大小为 n x n 的整数矩阵 grid 。

生成一个大小为 (n - 2) x (n - 2) 的整数矩阵  maxLocal ，并满足：

maxLocal[i][j] 等于 grid 中以 i + 1 行和 j + 1 列为中心的 3 x 3 矩阵中的 最大值 。
换句话说，我们希望找出 grid 中每个 3 x 3 矩阵中的最大值。

返回生成的矩阵。

 

示例 1：



输入：grid = [[9,9,8,1],[5,6,2,6],[8,2,6,4],[6,2,2,2]]
输出：[[9,9],[8,6]]
解释：原矩阵和生成的矩阵如上图所示。
注意，生成的矩阵中，每个值都对应 grid 中一个相接的 3 x 3 矩阵的最大值。
示例 2：



输入：grid = [[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,1],[1,1,1,1,1],[1,1,1,1,1]]
输出：[[2,2,2],[2,2,2],[2,2,2]]
解释：注意，2 包含在 grid 中每个 3 x 3 的矩阵中。
```

`思路`
遍历

`c# 实现`
```
public class Solution {
    public int[][] LargestLocal(int[][] grid) {
        int n = grid.Length;
        int[][] ans = new int[n-2][];
        for (int i = 0; i < n - 2; i++)
        {
            ans[i] = new int[n-2];
            for (int j = 0; j < n - 2; j++)
            {
                ans[i][j] = GetMaxNum(i,j,grid);
            }
           
        }
        return ans;
    }

    private int GetMaxNum(int row , int col ,int[][] grid){
        int max = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                max = Math.Max(max,grid[i+row][j+col]);
            }
        }
        return max;
    }
}
```

***

## 2023/2/28

## 2363. 合并相似的物品

[2363. 合并相似的物品](https://leetcode.cn/problems/merge-similar-items/description/)
```
给你两个二维整数数组 items1 和 items2 ，表示两个物品集合。每个数组 items 有以下特质：

items[i] = [valuei, weighti] 其中 valuei 表示第 i 件物品的 价值 ，weighti 表示第 i 件物品的 重量 。
items 中每件物品的价值都是 唯一的 。
请你返回一个二维数组 ret，其中 ret[i] = [valuei, weighti]， weighti 是所有价值为 valuei 物品的 重量之和 。

注意：ret 应该按价值 升序 排序后返回。

 

示例 1：

输入：items1 = [[1,1],[4,5],[3,8]], items2 = [[3,1],[1,5]]
输出：[[1,6],[3,9],[4,5]]
解释：
value = 1 的物品在 items1 中 weight = 1 ，在 items2 中 weight = 5 ，总重量为 1 + 5 = 6 。
value = 3 的物品再 items1 中 weight = 8 ，在 items2 中 weight = 1 ，总重量为 8 + 1 = 9 。
value = 4 的物品在 items1 中 weight = 5 ，总重量为 5 。
所以，我们返回 [[1,6],[3,9],[4,5]] 。
示例 2：

输入：items1 = [[1,1],[3,2],[2,3]], items2 = [[2,1],[3,2],[1,3]]
输出：[[1,4],[2,4],[3,4]]
解释：
value = 1 的物品在 items1 中 weight = 1 ，在 items2 中 weight = 3 ，总重量为 1 + 3 = 4 。
value = 2 的物品在 items1 中 weight = 3 ，在 items2 中 weight = 1 ，总重量为 3 + 1 = 4 。
value = 3 的物品在 items1 中 weight = 2 ，在 items2 中 weight = 2 ，总重量为 2 + 2 = 4 。
所以，我们返回 [[1,4],[2,4],[3,4]] 。
示例 3：

输入：items1 = [[1,3],[2,2]], items2 = [[7,1],[2,2],[1,4]]
输出：[[1,7],[2,4],[7,1]]
解释：
value = 1 的物品在 items1 中 weight = 3 ，在 items2 中 weight = 4 ，总重量为 3 + 4 = 7 。
value = 2 的物品在 items1 中 weight = 2 ，在 items2 中 weight = 2 ，总重量为 2 + 2 = 4 。
value = 7 的物品在 items2 中 weight = 1 ，总重量为 1 。
所以，我们返回 [[1,7],[2,4],[7,1]] 。
```

`思路`
哈希表
注意Ilist 要使用Sort 需要 ((List<IList<int>>) ans).Sort()

`c# 实现`
```
public class Solution {
    public IList<IList<int>> MergeSimilarItems(int[][] items1, int[][] items2) {
        Dictionary<int,int> dict = new Dictionary<int,int>();
        IList<IList<int>> ans = new List<IList<int>>();
        foreach (int[] item in items1)
        {
            dict.TryAdd(item[0],0);
            dict[item[0]] += item[1];
        }
        foreach (int[] item in items2)
        {
            dict.TryAdd(item[0],0);
            dict[item[0]] += item[1];
        }
        foreach (KeyValuePair<int, int> kvp in dict)
        {
            ans.Add(new List<int>(new int[]{kvp.Key,kvp.Value}));
        }
        ((List<IList<int>>) ans).Sort((a,b)=>{
            return a[0] - b[0];
        });
        return ans;
    }
}
```

***

## 2023/2/27

## 1144. 递减元素使数组呈锯齿状

[1144. 递减元素使数组呈锯齿状](https://leetcode.cn/problems/decrease-elements-to-make-array-zigzag/description/)
```
给你一个整数数组 nums，每次 操作 会从中选择一个元素并 将该元素的值减少 1。

如果符合下列情况之一，则数组 A 就是 锯齿数组：

每个偶数索引对应的元素都大于相邻的元素，即 A[0] > A[1] < A[2] > A[3] < A[4] > ...
或者，每个奇数索引对应的元素都大于相邻的元素，即 A[0] < A[1] > A[2] < A[3] > A[4] < ...
返回将数组 nums 转换为锯齿数组所需的最小操作次数。

 

示例 1：

输入：nums = [1,2,3]
输出：2
解释：我们可以把 2 递减到 0，或把 3 递减到 1。
示例 2：

输入：nums = [9,6,1,6,2]
输出：4
```

`思路`
贪心
分奇偶分别遍历数组，按比两边小的规则来计数操作数，因为奇偶都便利了，相当于自动考虑了比两边大的情况，最后返回操作数小的一个。

`c# 实现`
```
public class Solution {
    public int MovesToMakeZigzag(int[] nums) {
        return Math.Min(Help(nums, 0), Help(nums, 1));
    }

    public int Help(int[] nums, int pos) {
        int res = 0;
        for (int i = pos; i < nums.Length; i += 2) {
            int a = 0;
            if (i - 1 >= 0) {
                a = Math.Max(a, nums[i] - nums[i - 1] + 1);
            }
            if (i + 1 < nums.Length) {
                a = Math.Max(a, nums[i] - nums[i + 1] + 1);
            }
            res += a;
        }
        return res;
    }
}
```

***

## 2023/2/24

## 2357. 使数组中所有元素都等于零

[2357. 使数组中所有元素都等于零](https://leetcode.cn/problems/make-array-zero-by-subtracting-equal-amounts/description/)
```
给你一个非负整数数组 nums 。在一步操作中，你必须：

选出一个正整数 x ，x 需要小于或等于 nums 中 最小 的 非零 元素。
nums 中的每个正整数都减去 x。
返回使 nums 中所有元素都等于 0 需要的 最少 操作数。

 

示例 1：

输入：nums = [1,5,0,3,5]
输出：3
解释：
第一步操作：选出 x = 1 ，之后 nums = [0,4,0,2,4] 。
第二步操作：选出 x = 2 ，之后 nums = [0,2,0,0,2] 。
第三步操作：选出 x = 2 ，之后 nums = [0,0,0,0,0] 。
示例 2：

输入：nums = [0]
输出：0
解释：nums 中的每个元素都已经是 0 ，所以不需要执行任何操作。
```

`思路`
直接模拟

`c# 实现`
```
public class Solution {
    public int MinimumOperations(int[] nums) {
        int subSum = 0;
        int res = 0;
        Array.Sort(nums);
        for (int i = 0; i < nums.Length; i++)
        {
            int num = nums[i];
            num -= subSum;
            if (num <= 0)
            {
                continue;
            }
            subSum += num;
            res++;
        }
        return res;
    }
}
```

***

## 2023/2/23

## 1238. 循环码排列

[1238. 循环码排列](https://leetcode.cn/problems/circular-permutation-in-binary-representation/description/)
```
给你两个整数 n 和 start。你的任务是返回任意 (0,1,2,,...,2^n-1) 的排列 p，并且满足：

p[0] = start
p[i] 和 p[i+1] 的二进制表示形式只有一位不同
p[0] 和 p[2^n -1] 的二进制表示形式也只有一位不同
 

示例 1：

输入：n = 2, start = 3
输出：[3,2,0,1]
解释：这个排列的二进制表示是 (11,10,00,01)
     所有的相邻元素都有一位是不同的，另一个有效的排列是 [3,1,0,2]
示例 2：

输出：n = 3, start = 2
输出：[2,6,7,5,4,0,1,3]
解释：这个排列的二进制表示是 (010,110,111,101,100,000,001,011)
```

`思路`
这题需要格雷编码来构造，证明过程没看懂。。 只当练习位运算了。以后有需要再研究格雷编码吧

`c# 实现`
```
1.
public class Solution {
    public IList<int> CircularPermutation(int n, int start) {
        IList<int> ret = new List<int>();
        ret.Add(start);
        for (int i = 1; i <= n; i++) {
            int m = ret.Count;
            for (int j = m - 1; j >= 0; j--) {
                ret.Add(((ret[j] ^ start) | (1 << (i - 1))) ^ start);
            }
        }
        return ret;
    }
}

2.
public class Solution {
    public IList<int> CircularPermutation(int n, int start) {
        IList<int> ret = new List<int>();
        for (int i = 0; i < 1 << n; i++) {
            ret.Add((i >> 1) ^ i ^ start);
        }
        return ret;
    }
}
```

***

## 2023/2/22

## 1140. 石子游戏 II

[1140. 石子游戏 II](https://leetcode.cn/problems/stone-game-ii/description/)
```
爱丽丝和鲍勃继续他们的石子游戏。许多堆石子 排成一行，每堆都有正整数颗石子 piles[i]。游戏以谁手中的石子最多来决出胜负。

爱丽丝和鲍勃轮流进行，爱丽丝先开始。最初，M = 1。

在每个玩家的回合中，该玩家可以拿走剩下的 前 X 堆的所有石子，其中 1 <= X <= 2M。然后，令 M = max(M, X)。

游戏一直持续到所有石子都被拿走。

假设爱丽丝和鲍勃都发挥出最佳水平，返回爱丽丝可以得到的最大数量的石头。

 

示例 1：

输入：piles = [2,7,9,4,4]
输出：10
解释：如果一开始Alice取了一堆，Bob取了两堆，然后Alice再取两堆。爱丽丝可以得到2 + 4 + 4 = 10堆。如果Alice一开始拿走了两堆，那么Bob可以拿走剩下的三堆。在这种情况下，Alice得到2 + 7 = 9堆。返回10，因为它更大。
示例 2:

输入：piles = [1,2,3,4,5,100]
输出：104
```

`思路`
动态规划，设石堆总共n个，用dp[i][m] 来表示当选到第 i 堆石头时，M为m的情况下，最多能拿到多少个。

`c# 实现`
```
public class Solution {
    public int StoneGameII(int[] piles) {
        int n = piles.Length;
        int[][] dp = new int[n][];
        for (int i = 0; i < n; i++) {
            dp[i] = new int[n + 1];
            Array.Fill(dp[i], 0);
        }
        int sum = 0;
        for (int i = n - 1; i >= 0; i--)
        {
            sum += piles[i];
            for (int m = 1; m <= n; m++)
            {
                if (i+m*2 >= n)
                {
                    dp[i][m] = sum;
                }else{
                    for (int x = 1; x <= 2*m; x++)
                    {
                        dp[i][m] = Math.Max(dp[i][m],sum - dp[i+x][Math.Max(m,x)]);
                    }
                }
            }
        }
        return dp[0][1];
    }
}
```

***

## 2023/2/21

## 1326. 灌溉花园的最少水龙头数目

[1326. 灌溉花园的最少水龙头数目](https://leetcode.cn/problems/minimum-number-of-taps-to-open-to-water-a-garden/description/)
```
在 x 轴上有一个一维的花园。花园长度为 n，从点 0 开始，到点 n 结束。

花园里总共有 n + 1 个水龙头，分别位于 [0, 1, ..., n] 。

给你一个整数 n 和一个长度为 n + 1 的整数数组 ranges ，其中 ranges[i] （下标从 0 开始）表示：如果打开点 i 处的水龙头，可以灌溉的区域为 [i -  ranges[i], i + ranges[i]] 。

请你返回可以灌溉整个花园的 最少水龙头数目 。如果花园始终存在无法灌溉到的地方，请你返回 -1 。

 

示例 1：



输入：n = 5, ranges = [3,4,1,1,0,0]
输出：1
解释：
点 0 处的水龙头可以灌溉区间 [-3,3]
点 1 处的水龙头可以灌溉区间 [-3,5]
点 2 处的水龙头可以灌溉区间 [1,3]
点 3 处的水龙头可以灌溉区间 [2,4]
点 4 处的水龙头可以灌溉区间 [4,4]
点 5 处的水龙头可以灌溉区间 [5,5]
只需要打开点 1 处的水龙头即可灌溉整个花园 [0,5] 。
示例 2：

输入：n = 3, ranges = [0,0,0,0]
输出：-1
解释：即使打开所有水龙头，你也无法灌溉整个花园。
```

`思路`
贪心算法

`c# 实现`
```
public class Solution {
    public int MinTaps(int n, int[] ranges) {
        int[] rightMost = new int[n+1];
        for (int i = 0; i <= n; i++) {
            rightMost[i] = i;
        }
        for (int i = 0; i <= n; i++)
        {
            int start = Math.Max(0, i - ranges[i]);
            int end = Math.Min(n, i + ranges[i]);
            rightMost[start] = Math.Max(rightMost[start], end);
        }
        int res = 0, last = 0, pre = 0;
        for (int i = 0; i < n; i++)
        {
            last = Math.Max(rightMost[i] , last);
            if (i == last)
            {
                return -1;
            }
            if (i == pre)
            {
                res++;
                pre = last;
            }
        }
        return res;
    }
}
```

***

## 2023/2/20

## 2347. 最好的扑克手牌

[2347. 最好的扑克手牌](https://leetcode.cn/problems/best-poker-hand/description/)
```
给你一个整数数组 ranks 和一个字符数组 suit 。你有 5 张扑克牌，第 i 张牌大小为 ranks[i] ，花色为 suits[i] 。

下述是从好到坏你可能持有的 手牌类型 ：

"Flush"：同花，五张相同花色的扑克牌。
"Three of a Kind"：三条，有 3 张大小相同的扑克牌。
"Pair"：对子，两张大小一样的扑克牌。
"High Card"：高牌，五张大小互不相同的扑克牌。
请你返回一个字符串，表示给定的 5 张牌中，你能组成的 最好手牌类型 。

注意：返回的字符串 大小写 需与题目描述相同。

 

示例 1：

输入：ranks = [13,2,3,1,9], suits = ["a","a","a","a","a"]
输出："Flush"
解释：5 张扑克牌的花色相同，所以返回 "Flush" 。
示例 2：

输入：ranks = [4,4,2,4,4], suits = ["d","a","a","b","c"]
输出："Three of a Kind"
解释：第一、二和四张牌组成三张相同大小的扑克牌，所以得到 "Three of a Kind" 。
注意我们也可以得到 "Pair" ，但是 "Three of a Kind" 是更好的手牌类型。
有其他的 3 张牌也可以组成 "Three of a Kind" 手牌类型。
示例 3：

输入：ranks = [10,10,2,12,9], suits = ["a","b","c","a","d"]
输出："Pair"
解释：第一和第二张牌大小相同，所以得到 "Pair" 。
我们无法得到 "Flush" 或者 "Three of a Kind" 。
```

`思路`
哈希表逐个判断牌型

`c# 实现`
```
public class Solution {
    public string BestHand(int[] ranks, char[] suits) {
        for (int i = 1; i < suits.Length; i++)
        {
            if (suits[0] != suits[i])
            {
                break;
            }else{
                if (i == suits.Length - 1)
                {
                    return "Flush";
                }
            }
        }
        Dictionary<int,int> dict = new  Dictionary<int,int>();
        string ans = "High Card";
        for (int i = 0; i < ranks.Length; i++)
        {
            dict.TryAdd(ranks[i],0);
            dict[ranks[i]] ++;
            if (dict[ranks[i]] == 2)
            {
                ans = "Pair";
            }else if(dict[ranks[i]] > 2){
                ans = "Three of a Kind";
                break;
            }
        }
        return ans;
    }
}
```

***

## 2023/2/17

## 1139. 最大的以 1 为边界的正方形

[1139. 最大的以 1 为边界的正方形](https://leetcode.cn/problems/largest-1-bordered-square/description/)
```
给你一个由若干 0 和 1 组成的二维网格 grid，请你找出边界全部由 1 组成的最大 正方形 子网格，并返回该子网格中的元素数量。如果不存在，则返回 0。

 

示例 1：

输入：grid = [[1,1,1],[1,0,1],[1,1,1]]
输出：9
示例 2：

输入：grid = [[1,1,0,0]]
输出：1
```

`思路`
动态规划，用up , left 记录所有位置向左和向上的最大连续1长度

`c# 实现`
```
public class Solution {
    public int Largest1BorderedSquare(int[][] grid) {
        int m = grid.Length, n = grid[0].Length;
        int[][] left = new int[m + 1][];
        int[][] up = new int[m + 1][];
        for (int i = 0; i <= m; i++) {
            left[i] = new int[n + 1];
            up[i] = new int[n + 1];
        }
        int maxBorder = 0;
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                if (grid[i-1][j-1] == 1)
                {
                    left[i][j] = left[i][j-1] + 1;
                    up[i][j] = up[i-1][j] + 1;
                    int border = Math.Min(left[i][j], up[i][j]);
                    while(left[i - border +1][j] < border || up[i][j - border + 1] < border){
                        border --;
                    }
                    maxBorder = Math.Max(maxBorder, border);
                }
            }
        }
        return maxBorder * maxBorder;
    }
}
```

***

## 2023/2/16

## 2341. 数组能形成多少数对

[2341. 数组能形成多少数对](https://leetcode.cn/problems/maximum-number-of-pairs-in-array/description/)
```
给你一个下标从 0 开始的整数数组 nums 。在一步操作中，你可以执行以下步骤：

从 nums 选出 两个 相等的 整数
从 nums 中移除这两个整数，形成一个 数对
请你在 nums 上多次执行此操作直到无法继续执行。

返回一个下标从 0 开始、长度为 2 的整数数组 answer 作为答案，其中 answer[0] 是形成的数对数目，answer[1] 是对 nums 尽可能执行上述操作后剩下的整数数目。

 

示例 1：

输入：nums = [1,3,2,1,3,2,2]
输出：[3,1]
解释：
nums[0] 和 nums[3] 形成一个数对，并从 nums 中移除，nums = [3,2,3,2,2] 。
nums[0] 和 nums[2] 形成一个数对，并从 nums 中移除，nums = [2,2,2] 。
nums[0] 和 nums[1] 形成一个数对，并从 nums 中移除，nums = [2] 。
无法形成更多数对。总共形成 3 个数对，nums 中剩下 1 个数字。
示例 2：

输入：nums = [1,1]
输出：[1,0]
解释：nums[0] 和 nums[1] 形成一个数对，并从 nums 中移除，nums = [] 。
无法形成更多数对。总共形成 1 个数对，nums 中剩下 0 个数字。
示例 3：

输入：nums = [0]
输出：[0,1]
解释：无法形成数对，nums 中剩下 1 个数字。
```

`思路`
哈希表

`c# 实现`
```
public class Solution {
    public int[] NumberOfPairs(int[] nums) {
        int res = 0;
        HashSet<int> set = new HashSet<int>();
        for (int i = 0; i < nums.Length; i++)
        {
            if (set.Contains(nums[i]))
            {
                res++;
                set.Remove(nums[i]);
            }else{
                set.Add(nums[i]);
            }
        }
        return new int[2]{res,nums.Length - res*2};
    }
}
```

***

## 2023/2/14

## 1124. 表现良好的最长时间段

[1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/description/)
```
给你一份工作时间表 hours，上面记录着某一位员工每天的工作小时数。

我们认为当员工一天中的工作小时数大于 8 小时的时候，那么这一天就是「劳累的一天」。

所谓「表现良好的时间段」，意味在这段时间内，「劳累的天数」是严格 大于「不劳累的天数」。

请你返回「表现良好时间段」的最大长度。

 

示例 1：

输入：hours = [9,9,6,0,6,6,9]
输出：3
解释：最长的表现良好时间段是 [9,9,6]。
示例 2：

输入：hours = [6,6,6]
输出：0
```

`思路`
哈希表记录前缀值

`c# 实现`
```
public class Solution {
    public int LongestWPI(int[] hours) {
        int n = hours.Length;
        IDictionary<int, int> dictionary = new Dictionary<int, int>();
        int s = 0, res = 0;
        for (int i = 0; i < n; i++) {
            s += hours[i] > 8 ? 1 : -1;
            if (s > 0) {
                res = Math.Max(res, i + 1);
            } else {
                if (dictionary.ContainsKey(s - 1)) {
                    res = Math.Max(res, i - dictionary[s - 1]);
                }
            }
            if (!dictionary.ContainsKey(s)) {
                dictionary.Add(s, i);
            }
        }
        return res;
    }
}


```

***

## 2023/2/13

## 1234. 替换子串得到平衡字符串

[1234. 替换子串得到平衡字符串](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/description/)
```
有一个只含有 'Q', 'W', 'E', 'R' 四种字符，且长度为 n 的字符串。

假如在该字符串中，这四个字符都恰好出现 n/4 次，那么它就是一个「平衡字符串」。

 

给你一个这样的字符串 s，请通过「替换一个子串」的方式，使原字符串 s 变成一个「平衡字符串」。

你可以用和「待替换子串」长度相同的 任何 其他字符串来完成替换。

请返回待替换子串的最小可能长度。

如果原字符串自身就是一个平衡字符串，则返回 0。

 

示例 1：

输入：s = "QWER"
输出：0
解释：s 已经是平衡的了。
示例 2：

输入：s = "QQWE"
输出：1
解释：我们需要把一个 'Q' 替换成 'R'，这样得到的 "RQWE" (或 "QRWE") 是平衡的。
示例 3：

输入：s = "QQQW"
输出：2
解释：我们可以把前面的 "QQ" 替换成 "ER"。 
示例 4：

输入：s = "QQQQ"
输出：3
解释：我们可以替换后 3 个 'Q'，使 s = "QWER"。
```

`思路`
滑动窗口

`c# 实现`
```
 public class Solution {
    public int BalancedString(string s) {
         int[] cnt = new int[26];
        foreach (char c in s) {
            cnt[Idx(c)]++;
        }
        int n = s.Length;
        int partial = n / 4;
        if (Check(cnt, partial)) {
            return 0;
        }
        int res = n;
        for (int left = 0 , right = 0; left < n; left++)
        {
            while(right < n && !Check(cnt, partial)){
                cnt[Idx(s[right])]--;
                right ++;
            }
            if (!Check(cnt, partial)) {
                break;
            }
            res = Math.Min(res,right - left);
            cnt[Idx(s[left])]++;
        }
        return res;
    }

    public int Idx(char c) {
        return c - 'A';
    }

    public bool Check(int[] cnt, int partial) {
        if (cnt[Idx('Q')] > partial || cnt[Idx('W')] > partial || cnt[Idx('E')] > partial || cnt[Idx('R')] > partial) {
            return false;
        }
        return true;
    }

}
```

***

## 2023/2/9

## 1797. 设计一个验证系统

[1797. 设计一个验证系统](https://leetcode.cn/problems/design-authentication-manager/description/)
```
你需要设计一个包含验证码的验证系统。每一次验证中，用户会收到一个新的验证码，这个验证码在 currentTime 时刻之后 timeToLive 秒过期。如果验证码被更新了，那么它会在 currentTime （可能与之前的 currentTime 不同）时刻延长 timeToLive 秒。

请你实现 AuthenticationManager 类：

AuthenticationManager(int timeToLive) 构造 AuthenticationManager 并设置 timeToLive 参数。
generate(string tokenId, int currentTime) 给定 tokenId ，在当前时间 currentTime 生成一个新的验证码。
renew(string tokenId, int currentTime) 将给定 tokenId 且 未过期 的验证码在 currentTime 时刻更新。如果给定 tokenId 对应的验证码不存在或已过期，请你忽略该操作，不会有任何更新操作发生。
countUnexpiredTokens(int currentTime) 请返回在给定 currentTime 时刻，未过期 的验证码数目。
如果一个验证码在时刻 t 过期，且另一个操作恰好在时刻 t 发生（renew 或者 countUnexpiredTokens 操作），过期事件 优先于 其他操作。

 

示例 1：


输入：
["AuthenticationManager", "renew", "generate", "countUnexpiredTokens", "generate", "renew", "renew", "countUnexpiredTokens"]
[[5], ["aaa", 1], ["aaa", 2], [6], ["bbb", 7], ["aaa", 8], ["bbb", 10], [15]]
输出：
[null, null, null, 1, null, null, null, 0]

解释：
AuthenticationManager authenticationManager = new AuthenticationManager(5); // 构造 AuthenticationManager ，设置 timeToLive = 5 秒。
authenticationManager.renew("aaa", 1); // 时刻 1 时，没有验证码的 tokenId 为 "aaa" ，没有验证码被更新。
authenticationManager.generate("aaa", 2); // 时刻 2 时，生成一个 tokenId 为 "aaa" 的新验证码。
authenticationManager.countUnexpiredTokens(6); // 时刻 6 时，只有 tokenId 为 "aaa" 的验证码未过期，所以返回 1 。
authenticationManager.generate("bbb", 7); // 时刻 7 时，生成一个 tokenId 为 "bbb" 的新验证码。
authenticationManager.renew("aaa", 8); // tokenId 为 "aaa" 的验证码在时刻 7 过期，且 8 >= 7 ，所以时刻 8 的renew 操作被忽略，没有验证码被更新。
authenticationManager.renew("bbb", 10); // tokenId 为 "bbb" 的验证码在时刻 10 没有过期，所以 renew 操作会执行，该 token 将在时刻 15 过期。
authenticationManager.countUnexpiredTokens(15); // tokenId 为 "bbb" 的验证码在时刻 15 过期，tokenId 为 "aaa" 的验证码在时刻 7 过期，所有验证码均已过期，所以返回 0 。
```

`思路`
哈希表

`c# 实现`
```
public class AuthenticationManager {
    private int timeToLive;
    private Dictionary<string, int> dict;
    public AuthenticationManager(int timeToLive) {
        this.timeToLive = timeToLive;
        dict = new Dictionary<string, int>();
    }
    
    public void Generate(string tokenId, int currentTime) {
        dict.TryAdd(tokenId,0);
        dict[tokenId] = currentTime;
    }
    
    public void Renew(string tokenId, int currentTime) {
        if (!dict.ContainsKey(tokenId) )
        {
            return;
        }
        if (currentTime - dict[tokenId] >= timeToLive)
        {
            dict.Remove(tokenId);
            return;
        }
        dict[tokenId] = currentTime;
    }
    
    public int CountUnexpiredTokens(int currentTime) {
        foreach (KeyValuePair<string, int> kvp in dict)
        {
            if (currentTime - kvp.Value >= timeToLive)
            {
                dict.Remove(kvp.Key);
            }
        }
        return dict.Count;
    }
}

/**
 * Your AuthenticationManager object will be instantiated and called as such:
 * AuthenticationManager obj = new AuthenticationManager(timeToLive);
 * obj.Generate(tokenId,currentTime);
 * obj.Renew(tokenId,currentTime);
 * int param_3 = obj.CountUnexpiredTokens(currentTime);
 */
```

***

## 2023/2/8

## 1233. 删除子文件夹

[1233. 删除子文件夹](https://leetcode.cn/problems/remove-sub-folders-from-the-filesystem/description/)
```
你是一位系统管理员，手里有一份文件夹列表 folder，你的任务是要删除该列表中的所有 子文件夹，并以 任意顺序 返回剩下的文件夹。

如果文件夹 folder[i] 位于另一个文件夹 folder[j] 下，那么 folder[i] 就是 folder[j] 的 子文件夹 。

文件夹的「路径」是由一个或多个按以下格式串联形成的字符串：'/' 后跟一个或者多个小写英文字母。

例如，"/leetcode" 和 "/leetcode/problems" 都是有效的路径，而空字符串和 "/" 不是。
 

示例 1：

输入：folder = ["/a","/a/b","/c/d","/c/d/e","/c/f"]
输出：["/a","/c/d","/c/f"]
解释："/a/b" 是 "/a" 的子文件夹，而 "/c/d/e" 是 "/c/d" 的子文件夹。
示例 2：

输入：folder = ["/a","/a/b/c","/a/b/d"]
输出：["/a"]
解释：文件夹 "/a/b/c" 和 "/a/b/d" 都会被删除，因为它们都是 "/a" 的子文件夹。
示例 3：

输入: folder = ["/a/b/c","/a/b/ca","/a/b/d"]
输出: ["/a/b/c","/a/b/ca","/a/b/d"]
```

`思路`
排序+字典树

`c# 实现`
```
public class Solution {
    public IList<string> RemoveSubfolders(string[] folder) {
        Array.Sort(folder,(a,b)=>{
            return a.Length - b.Length;
        });
        IList<string> res = new List<string>();
        Trie tire = new Trie();
        for (int i = 0; i < folder.Length; i++)
        {
            if (folder[i] == " ")
            {
                continue;
            }
            Trie cur = tire;
            string[] sub = folder[i].Split('/');
            // for(int x = 0 ;x < sub.Length; x ++){
            //     res.Add(sub[x]);
            // }
            // return res;
            for (int j = 0; j < sub.Length; j++)
            {
                if(sub[j] == "")continue;
                if (cur.Children.ContainsKey("###"))
                {
                    goto cc;
                }
                if (cur.Children.ContainsKey(sub[j]))
                {
                    cur = cur.Children[sub[j]];
                    continue;
                }else{
                    cur.Children.Add(sub[j], new Trie());
                    cur = cur.Children[sub[j]];
                }
                
            }
            cur.Children.Add("###", new Trie());
            res.Add(folder[i]);
            cc : continue;
        }
        return res;
    }

    class Trie {
        public Dictionary<string, Trie> Children;
        public Trie() {
            Children = new Dictionary<string, Trie>();
        }
    }
}
```

***

## 2023/2/7

## 1604. 警告一小时内使用相同员工卡大于等于三次的人

[1604. 警告一小时内使用相同员工卡大于等于三次的人](https://leetcode.cn/problems/alert-using-same-key-card-three-or-more-times-in-a-one-hour-period/description/)
```
力扣公司的员工都使用员工卡来开办公室的门。每当一个员工使用一次他的员工卡，安保系统会记录下员工的名字和使用时间。如果一个员工在一小时时间内使用员工卡的次数大于等于三次，这个系统会自动发布一个 警告 。

给你字符串数组 keyName 和 keyTime ，其中 [keyName[i], keyTime[i]] 对应一个人的名字和他在 某一天 内使用员工卡的时间。

使用时间的格式是 24小时制 ，形如 "HH:MM" ，比方说 "23:51" 和 "09:49" 。

请你返回去重后的收到系统警告的员工名字，将它们按 字典序升序 排序后返回。

请注意 "10:00" - "11:00" 视为一个小时时间范围内，而 "23:51" - "00:10" 不被视为一小时内，因为系统记录的是某一天内的使用情况。

 

示例 1：

输入：keyName = ["daniel","daniel","daniel","luis","luis","luis","luis"], keyTime = ["10:00","10:40","11:00","09:00","11:00","13:00","15:00"]
输出：["daniel"]
解释："daniel" 在一小时内使用了 3 次员工卡（"10:00"，"10:40"，"11:00"）。
示例 2：

输入：keyName = ["alice","alice","alice","bob","bob","bob","bob"], keyTime = ["12:01","12:00","18:00","21:00","21:20","21:30","23:00"]
输出：["bob"]
解释："bob" 在一小时内使用了 3 次员工卡（"21:00"，"21:20"，"21:30"）。
示例 3：

输入：keyName = ["john","john","john"], keyTime = ["23:58","23:59","00:01"]
输出：[]
示例 4：

输入：keyName = ["leslie","leslie","leslie","clare","clare","clare","clare"], keyTime = ["13:00","13:20","14:00","18:00","18:51","19:30","19:49"]
输出：["clare","leslie"]
```

`思路`
哈希表记录每个同事所有开门时间，排序后统计一小时超过2次的人

`c# 实现`
```
public class Solution {
    public IList<string> AlertNames(string[] keyName, string[] keyTime) {
        int n = keyName.Length;
        IList<string> ans = new List<string>();
        Dictionary<string,List<int>> dict = new Dictionary<string,List<int>>();
        for (int i = 0; i < n; i++)
        {
            string name = keyName[i];
            string time = keyTime[i];
            dict.TryAdd(name,new List<int>());
            int hour = (time[0] - '0') * 10 + (time[1] - '0');
            int minute = (time[3] - '0') * 10 + (time[4] - '0');
            dict[name].Add(hour*60+minute);
        }
        foreach (KeyValuePair<string, List<int>> kvp in dict)
        {
            List<int> list = kvp.Value;
            list.Sort();
            for (int i = 2; i < list.Count; i++)
            {
                if (list[i] - list[i - 2] <= 60)
                {
                    ans.Add(kvp.Key);
                    break;
                }
            }
        }
        ((List<string>) ans).Sort();
        return ans;
    }
}
```

***

## 2023/2/6

## 2331. 计算布尔二叉树的值

[2331. 计算布尔二叉树的值](https://leetcode.cn/problems/evaluate-boolean-binary-tree/description/)
```
给你一棵 完整二叉树 的根，这棵树有以下特征：

叶子节点 要么值为 0 要么值为 1 ，其中 0 表示 False ，1 表示 True 。
非叶子节点 要么值为 2 要么值为 3 ，其中 2 表示逻辑或 OR ，3 表示逻辑与 AND 。
计算 一个节点的值方式如下：

如果节点是个叶子节点，那么节点的 值 为它本身，即 True 或者 False 。
否则，计算 两个孩子的节点值，然后将该节点的运算符对两个孩子值进行 运算 。
返回根节点 root 的布尔运算值。

完整二叉树 是每个节点有 0 个或者 2 个孩子的二叉树。

叶子节点 是没有孩子的节点。

 

示例 1：



输入：root = [2,1,3,null,null,0,1]
输出：true
解释：上图展示了计算过程。
AND 与运算节点的值为 False AND True = False 。
OR 运算节点的值为 True OR False = True 。
根节点的值为 True ，所以我们返回 true 。
示例 2：

输入：root = [0]
输出：false
解释：根节点是叶子节点，且值为 false，所以我们返回 false 。
```

`思路`
DFS遍历计算结果

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool EvaluateTree(TreeNode root) {
        if (root.val < 2)
        {
            return Convert.ToBoolean(root.val);
        }
        bool left = EvaluateTree(root.left);
        bool right = EvaluateTree(root.right);
        return root.val == 2 ? left || right : left && right;
    }
}

```

***

## 2023/2/3

## 1145. 二叉树着色游戏

[1145. 二叉树着色游戏](https://leetcode.cn/problems/binary-tree-coloring-game/description/)
```
有两位极客玩家参与了一场「二叉树着色」的游戏。游戏中，给出二叉树的根节点 root，树上总共有 n 个节点，且 n 为奇数，其中每个节点上的值从 1 到 n 各不相同。

最开始时：

「一号」玩家从 [1, n] 中取一个值 x（1 <= x <= n）；
「二号」玩家也从 [1, n] 中取一个值 y（1 <= y <= n）且 y != x。
「一号」玩家给值为 x 的节点染上红色，而「二号」玩家给值为 y 的节点染上蓝色。

之后两位玩家轮流进行操作，「一号」玩家先手。每一回合，玩家选择一个被他染过色的节点，将所选节点一个 未着色 的邻节点（即左右子节点、或父节点）进行染色（「一号」玩家染红色，「二号」玩家染蓝色）。

如果（且仅在此种情况下）当前玩家无法找到这样的节点来染色时，其回合就会被跳过。

若两个玩家都没有可以染色的节点时，游戏结束。着色节点最多的那位玩家获得胜利 ✌️。

现在，假设你是「二号」玩家，根据所给出的输入，假如存在一个 y 值可以确保你赢得这场游戏，则返回 true ；若无法获胜，就请返回 false 。

 
示例 1 ：


输入：root = [1,2,3,4,5,6,7,8,9,10,11], n = 11, x = 3
输出：true
解释：第二个玩家可以选择值为 2 的节点。
示例 2 ：

输入：root = [1,2,3], n = 3, x = 1
输出：false
```

`思路`
此题主要运用二叉树的DFS和BFS搜索

1.BFS找到x对应的节点

2.以x节点开始将树分为上，左，右三个部分。 DFS计算上，左，右的数量

3.满足上，左，右任意一个节点不少于 (n+1)/2 则可以获胜

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool BtreeGameWinningMove(TreeNode root, int n, int x) {
        // int total = CalNodeCnt(root);
        Queue<TreeNode> que = new Queue<TreeNode>();
        que.Enqueue(root);
        TreeNode find = null;
        while(que.Count > 0){
            find = que.Dequeue();
            if (find.left != null)
            {
                que.Enqueue(find.left);
            }
            if (find.right != null)
            {
                que.Enqueue(find.right);
            }
            if (find.val == x)
            {
                break;
            }
        }



        int left , right, parent = 0;
        left = CalNodeCnt(find.left);
        right = CalNodeCnt(find.right);
        parent = n - left - right - 1;
        int minWin = (n + 1)/2;
        return minWin <= left  || minWin <= right  || minWin <=  parent;
    }

    private int CalNodeCnt(TreeNode root){
        if (root == null)
        {
            return 0;
        }
        int cnt  = 1;
        cnt += CalNodeCnt(root.left);
        cnt += CalNodeCnt(root.right);

        return cnt;
    }
}
```

***

## 2023/2/1

## 2325. 解密消息

[2325. 解密消息](https://leetcode.cn/problems/decode-the-message/description/)
```
给你字符串 key 和 message ，分别表示一个加密密钥和一段加密消息。解密 message 的步骤如下：

使用 key 中 26 个英文小写字母第一次出现的顺序作为替换表中的字母 顺序 。
将替换表与普通英文字母表对齐，形成对照表。
按照对照表 替换 message 中的每个字母。
空格 ' ' 保持不变。
例如，key = "happy boy"（实际的加密密钥会包含字母表中每个字母 至少一次），据此，可以得到部分对照表（'h' -> 'a'、'a' -> 'b'、'p' -> 'c'、'y' -> 'd'、'b' -> 'e'、'o' -> 'f'）。
返回解密后的消息。

 

示例 1：



输入：key = "the quick brown fox jumps over the lazy dog", message = "vkbs bs t suepuv"
输出："this is a secret"
解释：对照表如上图所示。
提取 "the quick brown fox jumps over the lazy dog" 中每个字母的首次出现可以得到替换表。
示例 2：



输入：key = "eljuxhpwnyrdgtqkviszcfmabo", message = "zwx hnfx lqantp mnoeius ycgk vcnjrdb"
输出："the five boxing wizards jump quickly"
解释：对照表如上图所示。
提取 "eljuxhpwnyrdgtqkviszcfmabo" 中每个字母的首次出现可以得到替换表。
```

`思路`
模拟+哈希表记录

`c# 实现`
```
public class Solution {
    public string DecodeMessage(string key, string message) {
        Dictionary<char,char> dict = new Dictionary<char,char>();
        int cnt = 0;
        for (int i = 0; i < key.Length; i++)
        {
            if (key[i] != ' ' && !dict.ContainsKey(key[i]))
            {
                dict.TryAdd(key[i],(char)(cnt+'a'));
                cnt++;
            }
        }
        char[] ans = new char[message.Length];
        for (int i = 0; i < message.Length; i++)
        {
            ans[i] = message[i] == ' ' ? ' ' :  dict[message[i]];
        }
        return new String(ans);
    }
}
```

***

## 2023/1/31

## 2319. 判断矩阵是否是一个 X 矩阵

[2319. 判断矩阵是否是一个 X 矩阵](https://leetcode.cn/problems/check-if-matrix-is-x-matrix/description/)
```
如果一个正方形矩阵满足下述 全部 条件，则称之为一个 X 矩阵 ：

矩阵对角线上的所有元素都 不是 0
矩阵中所有其他元素都是 0
给你一个大小为 n x n 的二维整数数组 grid ，表示一个正方形矩阵。如果 grid 是一个 X 矩阵 ，返回 true ；否则，返回 false 。

 

示例 1：


输入：grid = [[2,0,0,1],[0,3,1,0],[0,5,2,0],[4,0,0,2]]
输出：true
解释：矩阵如上图所示。
X 矩阵应该满足：绿色元素（对角线上）都不是 0 ，红色元素都是 0 。
因此，grid 是一个 X 矩阵。
示例 2：


输入：grid = [[5,7,0],[0,3,1],[0,5,0]]
输出：false
解释：矩阵如上图所示。
X 矩阵应该满足：绿色元素（对角线上）都不是 0 ，红色元素都是 0 。
因此，grid 不是一个 X 矩阵。
```

`思路`
直接模拟

`c# 实现`
```
public class Solution {
    public bool CheckXMatrix(int[][] grid) {
        HashSet<string> set = new HashSet<string>();
        int n = grid.Length;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j || i + j == n - 1)
                {
                    if (grid[i][j] == 0)
                    {
                        return false;
                    }
                }else{
                    if (grid[i][j] != 0)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
```

***

## 2023/1/30

## 1669. 合并两个链表

[1669. 合并两个链表](https://leetcode.cn/problems/merge-in-between-linked-lists/description/)
```
给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。

请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。

下图中蓝色边和节点展示了操作后的结果：


请你返回结果链表的头指针。

 

示例 1：



输入：list1 = [0,1,2,3,4,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
输出：[0,1,2,1000000,1000001,1000002,5]
解释：我们删除 list1 中下标为 3 和 4 的两个节点，并将 list2 接在该位置。上图中蓝色的边和节点为答案链表。
示例 2：


输入：list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
输出：[0,1,1000000,1000001,1000002,1000003,1000004,6]
解释：上图中蓝色的边和节点为答案链表。
```

`思路`
找到对应节点，修改指针指向

`c# 实现`
```
public class Solution {
    public ListNode MergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        ListNode cur = list1;
        ListNode tail = null;
        ListNode head = null;
        for (int i = 0; i < b; i++)
        {
            if (i == a - 1)
            {
                head = cur;
            }
            cur = cur.next;
        }
        tail = cur.next;
        head.next = list2;
        while(list2.next != null){
            list2 = list2.next;
        }
        list2.next = tail;
        return list1;
    }
}
```

***


## 2023/1/17

## 1814. 统计一个数组中好对子的数目

[1814. 统计一个数组中好对子的数目](https://leetcode.cn/problems/count-nice-pairs-in-an-array/description/)
```
给你一个数组 nums ，数组中只包含非负整数。定义 rev(x) 的值为将整数 x 各个数字位反转得到的结果。比方说 rev(123) = 321 ， rev(120) = 21 。我们称满足下面条件的下标对 (i, j) 是 好的 ：

0 <= i < j < nums.length
nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])
请你返回好下标对的数目。由于结果可能会很大，请将结果对 109 + 7 取余 后返回。

 

示例 1：

输入：nums = [42,11,1,97]
输出：2
解释：两个坐标对为：
 - (0,3)：42 + rev(97) = 42 + 79 = 121, 97 + rev(42) = 97 + 24 = 121 。
 - (1,2)：11 + rev(1) = 11 + 1 = 12, 1 + rev(11) = 1 + 11 = 12 。
示例 2：

输入：nums = [13,10,35,24,76]
输出：4
```

`思路`
反转后的数和原数字的差相同的数字则满足好的数，遍历找出所有好的。

`c# 实现`
```
public class Solution {
    const int MOD = 1000000007;
    public int CountNicePairs(int[] nums) {
        int n = nums.Length;
        int res = 0;
        Dictionary<int , int> dict = new Dictionary <int , int>();
        for (int i = 0; i < n; i++)
        {
            int diff = Reverse(nums[i]) - nums[i];
            dict.TryAdd(diff,0);
            res = (res + dict[diff])%MOD;
            dict[diff] ++;
        }
    
        return res;
    }

    public int Reverse(int x)
    {

        int num = 0;
        while (x > 0) {
            num = num * 10 + x % 10;
            x /= 10;
        }
        return num;
    }
}
```

***

## 2023/1/16

## 1813. 句子相似性 III

[1813. 句子相似性 III](https://leetcode.cn/problems/sentence-similarity-iii/description/)
```
一个句子是由一些单词与它们之间的单个空格组成，且句子的开头和结尾没有多余空格。比方说，"Hello World" ，"HELLO" ，"hello world hello world" 都是句子。每个单词都 只 包含大写和小写英文字母。

如果两个句子 sentence1 和 sentence2 ，可以通过往其中一个句子插入一个任意的句子（可以是空句子）而得到另一个句子，那么我们称这两个句子是 相似的 。比方说，sentence1 = "Hello my name is Jane" 且 sentence2 = "Hello Jane" ，我们可以往 sentence2 中 "Hello" 和 "Jane" 之间插入 "my name is" 得到 sentence1 。

给你两个句子 sentence1 和 sentence2 ，如果 sentence1 和 sentence2 是相似的，请你返回 true ，否则返回 false 。

 

示例 1：

输入：sentence1 = "My name is Haley", sentence2 = "My Haley"
输出：true
解释：可以往 sentence2 中 "My" 和 "Haley" 之间插入 "name is" ，得到 sentence1 。
示例 2：

输入：sentence1 = "of", sentence2 = "A lot of words"
输出：false
解释：没法往这两个句子中的一个句子只插入一个句子就得到另一个句子。
示例 3：

输入：sentence1 = "Eating right now", sentence2 = "Eating"
输出：true
解释：可以往 sentence2 的结尾插入 "right now" 得到 sentence1 。
示例 4：

输入：sentence1 = "Luky", sentence2 = "Lucccky"
输出：false

```

`思路`
分割字符串后 找规律

`c# 实现`
```
public class Solution {
    public bool AreSentencesSimilar(string sentence1, string sentence2) {
        string[] words1 = sentence1.Split(new char[]{' '});
        string[] words2 = sentence2.Split(new char[]{' '});
        int i = 0, j = 0;
        while (i < words1.Length && i < words2.Length && words1[i].Equals(words2[i])) {
            i++;
        }
        while (j < words1.Length - i && j < words2.Length - i && words1[words1.Length - j - 1].Equals(words2[words2.Length - j - 1])) {
            j++;
        }
        return i + j == Math.Min(words1.Length, words2.Length);
    }
}
```

***

## 2023/1/13

## 2287. 重排字符形成目标字符串

[2287. 重排字符形成目标字符串](https://leetcode.cn/problems/rearrange-characters-to-make-target-string/description/)
```
给你两个下标从 0 开始的字符串 s 和 target 。你可以从 s 取出一些字符并将其重排，得到若干新的字符串。

从 s 中取出字符并重新排列，返回可以形成 target 的 最大 副本数。

 

示例 1：

输入：s = "ilovecodingonleetcode", target = "code"
输出：2
解释：
对于 "code" 的第 1 个副本，选取下标为 4 、5 、6 和 7 的字符。
对于 "code" 的第 2 个副本，选取下标为 17 、18 、19 和 20 的字符。
形成的字符串分别是 "ecod" 和 "code" ，都可以重排为 "code" 。
可以形成最多 2 个 "code" 的副本，所以返回 2 。
示例 2：

输入：s = "abcba", target = "abc"
输出：1
解释：
选取下标为 0 、1 和 2 的字符，可以形成 "abc" 的 1 个副本。 
可以形成最多 1 个 "abc" 的副本，所以返回 1 。
注意，尽管下标 3 和 4 分别有额外的 'a' 和 'b' ，但不能重用下标 2 处的 'c' ，所以无法形成 "abc" 的第 2 个副本。
示例 3：

输入：s = "abbaccaddaeea", target = "aaaaa"
输出：1
解释：
选取下标为 0 、3 、6 、9 和 12 的字符，可以形成 "aaaaa" 的 1 个副本。
可以形成最多 1 个 "aaaaa" 的副本，所以返回 1 。
```

`思路`
用两个哈希表分别记录两个字符串的所有字符数量，比较能构成几个target

`c# 实现`
```
public class Solution {
    public int RearrangeCharacters(string s, string target) {
        Dictionary<char , int> targetDict = new Dictionary<char , int>();
        Dictionary<char , int> sDict = new Dictionary<char , int>();
        foreach (char c in target)
        {
            targetDict.TryAdd(c,0);
            targetDict[c] ++;
        }
        foreach (char c in s)
        {
            sDict.TryAdd(c,0);
            sDict[c] ++;
        }
        int res = int.MaxValue;
        foreach (KeyValuePair<char, int> pair in targetDict)
        {
            char c = pair.Key;
            int cnt = pair.Value;
            int totalCnt = sDict.ContainsKey(c) ? sDict[c] : 0;
            res = Math.Min(res , totalCnt / cnt);
            if (res == 0)
            {
                return 0;
            }
        }
        return res;
    }
}
```

***

## 2023/1/12

## 1807. 替换字符串中的括号内容

[1807. 替换字符串中的括号内容](https://leetcode.cn/problems/evaluate-the-bracket-pairs-of-a-string/description/)
```
给你一个字符串 s ，它包含一些括号对，每个括号中包含一个 非空 的键。

比方说，字符串 "(name)is(age)yearsold" 中，有 两个 括号对，分别包含键 "name" 和 "age" 。
你知道许多键对应的值，这些关系由二维字符串数组 knowledge 表示，其中 knowledge[i] = [keyi, valuei] ，表示键 keyi 对应的值为 valuei 。

你需要替换 所有 的括号对。当你替换一个括号对，且它包含的键为 keyi 时，你需要：

将 keyi 和括号用对应的值 valuei 替换。
如果从 knowledge 中无法得知某个键对应的值，你需要将 keyi 和括号用问号 "?" 替换（不需要引号）。
knowledge 中每个键最多只会出现一次。s 中不会有嵌套的括号。

请你返回替换 所有 括号对后的结果字符串。

 

示例 1：

输入：s = "(name)is(age)yearsold", knowledge = [["name","bob"],["age","two"]]
输出："bobistwoyearsold"
解释：
键 "name" 对应的值为 "bob" ，所以将 "(name)" 替换为 "bob" 。
键 "age" 对应的值为 "two" ，所以将 "(age)" 替换为 "two" 。
示例 2：

输入：s = "hi(name)", knowledge = [["a","b"]]
输出："hi?"
解释：由于不知道键 "name" 对应的值，所以用 "?" 替换 "(name)" 。
示例 3：

输入：s = "(a)(a)(a)aaa", knowledge = [["a","yes"]]
输出："yesyesyesaaa"
解释：相同的键在 s 中可能会出现多次。
键 "a" 对应的值为 "yes" ，所以将所有的 "(a)" 替换为 "yes" 。
注意，不在括号里的 "a" 不需要被替换。
```

`思路`
哈希表记录

`c# 实现`
```
public class Solution {
    public string Evaluate(string s, IList<IList<string>> knowledge) {
        Dictionary<string,string> dict = new Dictionary<string,string>();
        for (int i = 0; i < knowledge.Count; i++)
        {
            dict.TryAdd(knowledge[i][0],knowledge[i][1]);
        }

        StringBuilder key = new StringBuilder();
        StringBuilder res = new StringBuilder();
        bool addKey = false;
        foreach (char c in s)
        {
            if (c == '(')
            {
                addKey = true;
            }else if(c == ')'){
                if (dict.ContainsKey(key.ToString()))
                {
                    res.Append(dict[key.ToString()]);
                }else{
                    res.Append('?');
                }
                addKey = false;
                key.Length = 0;
            }else{
                if (addKey)
                {
                    key.Append(c);
                }else{
                    res.Append(c);
                }
            }
        }
        return res.ToString();
    }
}
```

***

## 2023/1/11

## 2283. 判断一个数的数字计数是否等于数位的值

[2283. 判断一个数的数字计数是否等于数位的值](https://leetcode.cn/problems/check-if-number-has-equal-digit-count-and-digit-value/description/)
```
给你一个下标从 0 开始长度为 n 的字符串 num ，它只包含数字。

如果对于 每个 0 <= i < n 的下标 i ，都满足数位 i 在 num 中出现了 num[i]次，那么请你返回 true ，否则返回 false 。

 

示例 1：

输入：num = "1210"
输出：true
解释：
num[0] = '1' 。数字 0 在 num 中出现了一次。
num[1] = '2' 。数字 1 在 num 中出现了两次。
num[2] = '1' 。数字 2 在 num 中出现了一次。
num[3] = '0' 。数字 3 在 num 中出现了零次。
"1210" 满足题目要求条件，所以返回 true 。
示例 2：

输入：num = "030"
输出：false
解释：
num[0] = '0' 。数字 0 应该出现 0 次，但是在 num 中出现了一次。
num[1] = '3' 。数字 1 应该出现 3 次，但是在 num 中出现了零次。
num[2] = '0' 。数字 2 在 num 中出现了 0 次。
下标 0 和 1 都违反了题目要求，所以返回 false 。
```

`思路`
哈希表记录

`c# 实现`
```
public class Solution {
    public bool DigitCount(string num) {
        Dictionary<int,int> dict = new  Dictionary<int,int>();
        for (int i = 0; i < num.Length; i++)
        {
            int digit = num[i] - '0';
            dict.TryAdd(digit,0);
            dict[digit] ++;
        }
        for (int i = 0; i < num.Length; i++)
        {
            int digit = num[i] - '0';
            int cnt = dict.ContainsKey(i) ? dict[i] : 0;
            if (cnt != digit)
            {
                return false;
            }
        }
        return true;
    }
}
```

***

## 2023/1/9

## 1806. 还原排列的最少操作步数

[1806. 还原排列的最少操作步数](https://leetcode.cn/problems/minimum-number-of-operations-to-reinitialize-a-permutation/description/)
```
给你一个偶数 n​​​​​​ ，已知存在一个长度为 n 的排列 perm ，其中 perm[i] == i​（下标 从 0 开始 计数）。

一步操作中，你将创建一个新数组 arr ，对于每个 i ：

如果 i % 2 == 0 ，那么 arr[i] = perm[i / 2]
如果 i % 2 == 1 ，那么 arr[i] = perm[n / 2 + (i - 1) / 2]
然后将 arr​​ 赋值​​给 perm 。

要想使 perm 回到排列初始值，至少需要执行多少步操作？返回最小的 非零 操作步数。

 

示例 1：

输入：n = 2
输出：1
解释：最初，perm = [0,1]
第 1 步操作后，perm = [0,1]
所以，仅需执行 1 步操作
示例 2：

输入：n = 4
输出：2
解释：最初，perm = [0,1,2,3]
第 1 步操作后，perm = [0,2,1,3]
第 2 步操作后，perm = [0,1,2,3]
所以，仅需执行 2 步操作
示例 3：

输入：n = 6
输出：4
```

`思路`
直接模拟，直到变化的数组与最初数组相等。

`c# 实现`
```
public class Solution {
    public int ReinitializePermutation(int n) {
        int[] stander = new int[n];
        for(int i = 0; i < n; i++){
            stander[i] = i;
        }
        int res = 1;
        int[] cur = cal(stander);
        while(!Enumerable.SequenceEqual(cur, stander)){
            cur = cal(cur);
            res ++;
        }
        return res;
    }

    private int[] cal(int[] arr){
        int n = arr.Length;
        int[] res = new int[n];
        for(int i =0 ; i < n; i++){
            if((i & 1) == 0){
                res[i] = arr[i/2];
            }else{
                res[i] = arr[n / 2 + (i - 1)/2];
            }
        }
        return res;
    }
}
```

***

## 2023/1/7

## 1658. 将 x 减到 0 的最小操作数

[1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/description/)
```
给你一个整数数组 nums 和一个整数 x 。每一次操作时，你应当移除数组 nums 最左边或最右边的元素，然后从 x 中减去该元素的值。请注意，需要 修改 数组以供接下来的操作使用。

如果可以将 x 恰好 减到 0 ，返回 最小操作数 ；否则，返回 -1 。

 

示例 1：

输入：nums = [1,1,4,2,3], x = 5
输出：2
解释：最佳解决方案是移除后两个元素，将 x 减到 0 。
示例 2：

输入：nums = [5,6,7,8,9], x = 4
输出：-1
示例 3：

输入：nums = [3,2,20,1,1,3], x = 10
输出：5
解释：最佳解决方案是移除后三个元素和前两个元素（总共 5 次操作），将 x 减到 0 。
```

`思路`
直接遍历会超时，贴一个官方题解 滑动窗口，巧妙的是rsum一开始是最大值，一次便利就可以包括left和right的全部情况

`c# 实现`
```
public class Solution {
    public int MinOperations(int[] nums, int x) {
        int n = nums.Length;
        int res = -1;
        for (int i = 0; i < n + 1; i++)
        {   
            int left = 0;
            if (i > 0)
            {
                for (int i2 = 0; i2 < i; i2++)
                {
                    left += nums[i2];
                    if (left == x)
                    {
                        if (res == -1)
                        {
                            res = i2+1;
                        }else{
                            res = Math.Min(i2+1,res);
                        }
                    }else if (left > x)
                    {
                        goto cc;
                    }
                }
            }
            for (int j = 1; j <= n - i; j++)
            {
                int right = 0;
                for (int j2 = 0; j2 < j; j2++)
                {
                    right += nums[n - j2 - 1];
                    if (left + right == x)
                    {
                        if (res == -1)
                        {
                            res = i + j2 + 1;
                        }else{
                            res = Math.Min(i + j2 + 1,res);
                        }
                    }else if (left + right > x)
                    {
                        goto cc;
                    }
                }
            }
            cc : continue;
        }
        return res;
    }
}

官方题解 滑动窗口
public class Solution {
    public int MinOperations(int[] nums, int x) {
        int n = nums.Length;
        int sum = nums.Sum();

        if (sum < x) {
            return -1;
        }

        int right = 0;
        int lsum = 0, rsum = sum;
        int ans = n + 1;

        for (int left = -1; left < n; left++)
        {
            if (left != -1)
            {
                lsum += nums[left];
            }
            while(right < n && lsum + rsum > x){
                rsum -= nums[right];
                right ++;
            }
            if (lsum + rsum == x)
            {
                ans = Math.Min(ans,left+1+n-right);
            }
        }

        return ans > n ? -1 : ans;
    }
}

```

***

## 2023/1/6

## 2180. 统计各位数字之和为偶数的整数个数

[2180. 统计各位数字之和为偶数的整数个数](https://leetcode.cn/problems/count-integers-with-even-digit-sum/description/)
```
给你一个正整数 num ，请你统计并返回 小于或等于 num 且各位数字之和为 偶数 的正整数的数目。

正整数的 各位数字之和 是其所有位上的对应数字相加的结果。

 

示例 1：

输入：num = 4
输出：2
解释：
只有 2 和 4 满足小于等于 4 且各位数字之和为偶数。    
示例 2：

输入：num = 30
输出：14
解释：
只有 14 个整数满足小于等于 30 且各位数字之和为偶数，分别是： 
2、4、6、8、11、13、15、17、19、20、22、24、26 和 28 。
```

`思路`
寻找规律，每10个数有5个

`c# 实现`
```
public class Solution {
    public int CountEven(int num) {
        int ten = num / 10;             
        int res = ten * 5;              
        int digit = num - ten*10;       
        int front = 0;
        while(ten > 0){
            front += ten % 10;
            ten /= 10;
        }
        if (((front) & 1) == 0)
        {
            res += digit/2+1;
        }else{
            res += (digit+1)/2;
        }
        return res - 1;
    }
}
```

***

## 2023/1/5

## 1802. 有界数组中指定下标处的最大值

[1802. 有界数组中指定下标处的最大值](https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/description/)
```
给你三个正整数 n、index 和 maxSum 。你需要构造一个同时满足下述所有条件的数组 nums（下标 从 0 开始 计数）：

nums.length == n
nums[i] 是 正整数 ，其中 0 <= i < n
abs(nums[i] - nums[i+1]) <= 1 ，其中 0 <= i < n-1
nums 中所有元素之和不超过 maxSum
nums[index] 的值被 最大化
返回你所构造的数组中的 nums[index] 。

注意：abs(x) 等于 x 的前提是 x >= 0 ；否则，abs(x) 等于 -x 。

 

示例 1：

输入：n = 4, index = 2,  maxSum = 6
输出：2
解释：数组 [1,1,2,1] 和 [1,2,2,1] 满足所有条件。不存在其他在指定下标处具有更大值的有效数组。
示例 2：

输入：n = 6, index = 1,  maxSum = 10
输出：3
```

`思路`
假设第index个数为x，x一定为[1,maxSum],可以用二分查找x，当x满足x+左边的和+右边的和 <= maxSum时，则继续往右查找，否则向左查找

`c# 实现`
```
public class Solution {
    public int MaxValue(int n, int index, int maxSum) {
        int left = 1, right = maxSum;
        while(left < right){
            int mid = (left + right + 1)/2;
            if (checkVaild(mid,n,index,maxSum))
            {
                left = mid;
            }else {
                right = mid - 1;
            }
        }
        return left;
    }

    public bool checkVaild(int mid , int n , int index , int maxSum){
        int left = index;
        int right = n - index - 1;
        return mid + Cal(mid , left) + Cal(mid , right) <= maxSum;
    }

    public long Cal(int big, int len){
        if (len > big - 1)
        {
            int ones = len - (big - 1);
            return (long)big * (big - 1) / 2 + ones;
        }else{
            int small = big - len;
            return (long)(big - 1 + small)*len / 2; 
        }
    }
}
```

***

## 2023/1/3

## 2042. 检查句子中的数字是否递增

[2042. 检查句子中的数字是否递增](https://leetcode.cn/problems/check-if-numbers-are-ascending-in-a-sentence/submissions/392616167/)
```
句子是由若干 token 组成的一个列表，token 间用 单个 空格分隔，句子没有前导或尾随空格。每个 token 要么是一个由数字 0-9 组成的不含前导零的 正整数 ，要么是一个由小写英文字母组成的 单词 。
* 示例，"a puppy has 2 eyes 4 legs" 是一个由 7 个 token 组成的句子："2" 和 "4" 是数字，其他像 "puppy" 这样的 tokens 属于单词。
给你一个表示句子的字符串 s ，你需要检查 s 中的 全部 数字是否从左到右严格递增（即，除了最后一个数字，s 中的 每个 数字都严格小于它 右侧 的数字）。
如果满足题目要求，返回 true ，否则，返回 false 。
 
示例 1：
￼
输入：s = "1 box has 3 blue 4 red 6 green and 12 yellow marbles"
输出：true
解释：句子中的数字是：1, 3, 4, 6, 12 。
这些数字是按从左到右严格递增的 1 < 3 < 4 < 6 < 12 。
示例 2：
输入：s = "hello world 5 x 5"
输出：false
解释：句子中的数字是：5, 5 。这些数字不是严格递增的。
示例 3：
￼
输入：s = "sunset is at 7 51 pm overnight lows will be in the low 50 and 60 s"
输出：false
解释：s 中的数字是：7, 51, 50, 60 。这些数字不是严格递增的。
示例 4：
输入：s = "4 5 11 26"
输出：true
解释：s 中的数字是：4, 5, 11, 26 。
这些数字是按从左到右严格递增的：4 < 5 < 11 < 26 。
```

`思路`
分割字符串，比较数字是否递增

`c# 实现`
```
public class Solution {
    public bool AreNumbersAscending(string s) {
        string[] arr = s.Split(' ');
        int num = 0;
        for (int i = 0; i < arr.Length; i++)
        {
            int temp = 0;
            bool res = int.TryParse(arr[i], out temp);
            if (res)
            {
                if (temp <= num)
                {
                    return false;
                }else{
                    num = temp;
                }
            }
        }
        return true;
    }
}
```

***

## 2022/12/30

## 855. 考场就座

[855. 考场就座](https://leetcode.cn/problems/exam-room/description/)
```
在考场里，一排有 N 个座位，分别编号为 0, 1, 2, ..., N-1 。

当学生进入考场后，他必须坐在能够使他与离他最近的人之间的距离达到最大化的座位上。如果有多个这样的座位，他会坐在编号最小的座位上。(另外，如果考场里没有人，那么学生就坐在 0 号座位上。)

返回 ExamRoom(int N) 类，它有两个公开的函数：其中，函数 ExamRoom.seat() 会返回一个 int （整型数据），代表学生坐的位置；函数 ExamRoom.leave(int p) 代表坐在座位 p 上的学生现在离开了考场。每次调用 ExamRoom.leave(p) 时都保证有学生坐在座位 p 上。

 

示例：

输入：["ExamRoom","seat","seat","seat","seat","leave","seat"], [[10],[],[],[],[],[4],[]]
输出：[null,0,9,4,2,null,5]
解释：
ExamRoom(10) -> null
seat() -> 0，没有人在考场里，那么学生坐在 0 号座位上。
seat() -> 9，学生最后坐在 9 号座位上。
seat() -> 4，学生最后坐在 4 号座位上。
seat() -> 2，学生最后坐在 2 号座位上。
leave(4) -> null
seat() -> 5，学生最后坐在 5 号座位上。
```

`思路`
直接模拟

`c# 实现`
```
public class ExamRoom {
    SortedList list;
    int capacity = 0;
    public ExamRoom(int n) {
        list = new SortedList();
        capacity = n;
    }
    
    public int Seat() {
        if(list.Count == 0){
            list.Add(0,0);
            return 0;
        }

        int first = (int)list.GetKey(0) - 0;
        int last = capacity - (int)list.GetKey(list.Count - 1) - 1;

        int max = 0;
        int pos = 0;

         

        for(int i = 1; i < list.Count; i ++){
            int diff = ((int)list.GetKey(i) - (int)list.GetKey(i - 1)) / 2;
            if(diff > max){
                pos = (int)list.GetKey(i - 1) + diff;
                max = diff;
            }
        }
        
        if(first >= max || last > max){
            if(first < last){
                list.Add(capacity - 1,capacity-1);
                return capacity-1;
            }else{
                list.Add(0,0);
                return 0;
            }
        }else{
            list.Add(pos,pos);
            return pos;
        }
        // return 0;
    }
    
    public void Leave(int p) {
        list.Remove(p);
    }
}

/**
 * Your ExamRoom object will be instantiated and called as such:
 * ExamRoom obj = new ExamRoom(n);
 * int param_1 = obj.Seat();
 * obj.Leave(p);
 */
```

***

## 2022/12/29

## 2032. 至少在两个数组中出现的值

[2032. 至少在两个数组中出现的值](https://leetcode.cn/problems/two-out-of-three/description/)
```
给你三个整数数组 nums1、nums2 和 nums3 ，请你构造并返回一个 元素各不相同的 数组，且由 至少 在 两个 数组中出现的所有值组成。数组中的元素可以按 任意 顺序排列。
 

示例 1：

输入：nums1 = [1,1,3,2], nums2 = [2,3], nums3 = [3]
输出：[3,2]
解释：至少在两个数组中出现的所有值为：
- 3 ，在全部三个数组中都出现过。
- 2 ，在数组 nums1 和 nums2 中出现过。
示例 2：

输入：nums1 = [3,1], nums2 = [2,3], nums3 = [1,2]
输出：[2,3,1]
解释：至少在两个数组中出现的所有值为：
- 2 ，在数组 nums2 和 nums3 中出现过。
- 3 ，在数组 nums1 和 nums2 中出现过。
- 1 ，在数组 nums1 和 nums3 中出现过。
示例 3：

输入：nums1 = [1,2,2], nums2 = [4,3,3], nums3 = [5]
输出：[]
解释：不存在至少在两个数组中出现的值。
```

`思路`
记录各个数组重复的值

`c# 实现`
```
public class Solution {
    public IList<int> TwoOutOfThree(int[] nums1, int[] nums2, int[] nums3) {
        IList<int> ans = new List<int>();
        ISet<int> hset = new HashSet<int>();
        ISet<int> hset2 = new HashSet<int>();
        for(int i = 0; i < nums1.Length; i ++){
             hset.Add(nums1[i]);
        }

        for(int i = 0; i < nums2.Length; i ++){
            if(!hset2.Contains(nums2[i])){
                hset2.Add(nums2[i]);
                if(hset.Contains(nums2[i])&& !ans.Contains(nums2[i])){
                    ans.Add(nums2[i]);
                }
                hset.Add(nums2[i]);
            }
        }
        //  return ans;
        hset2.Clear();
        for(int i = 0; i < nums3.Length; i ++){
            if(!hset2.Contains(nums3[i])){
                hset2.Add(nums3[i]);
                if(hset.Contains(nums3[i])&& !ans.Contains(nums3[i])){
                    ans.Add(nums3[i]);
                }
                hset.Add(nums3[i]);
            }
        }
        return ans;

    }
}
```

***

## 2022/12/26

## 1759. 统计同构子字符串的数目

[1759. 统计同构子字符串的数目](https://leetcode.cn/problems/count-number-of-homogenous-substrings/description/)
```
给你一个字符串 s ，返回 s 中 同构子字符串 的数目。由于答案可能很大，只需返回对 109 + 7 取余 后的结果。

同构字符串 的定义为：如果一个字符串中的所有字符都相同，那么该字符串就是同构字符串。

子字符串 是字符串中的一个连续字符序列。

 

示例 1：

输入：s = "abbcccaa"
输出：13
解释：同构子字符串如下所列：
"a"   出现 3 次。
"aa"  出现 1 次。
"b"   出现 2 次。
"bb"  出现 1 次。
"c"   出现 3 次。
"cc"  出现 2 次。
"ccc" 出现 1 次。
3 + 1 + 2 + 1 + 3 + 2 + 1 = 13
示例 2：

输入：s = "xy"
输出：2
解释：同构子字符串是 "x" 和 "y" 。
示例 3：

输入：s = "zzzzz"
输出：15
```

`思路`
遍历字符串，计数连续字符个数，然后将答案相加

`c# 实现`
```
public class Solution {
    const int MOD = 1000000007;
    public int CountHomogenous(string s) {
        long res = 0;
        for(int i = 0; i < s.Length; i++){
            int cnt = 1;
            while(i+cnt < s.Length && s[i] == s[i+cnt]){
                cnt ++;
            }
            res += display(cnt);
            i += (cnt-1);
        }
        return (int)(res%MOD);
    }

    public long display(int n) {
        long res = 1;
        while (n != 1) {
            res = (long)res + n;
            n = n - 1;
        }
        return res;
    }
}
```

***

## 2022/12/23

## 2011. 执行操作后的变量值

[2011. 执行操作后的变量值](https://leetcode.cn/problems/final-value-of-variable-after-performing-operations/description/)
```
存在一种仅支持 4 种操作和 1 个变量 X 的编程语言：
* ++X 和 X++ 使变量 X 的值 加 1
* --X 和 X-- 使变量 X 的值 减 1
最初，X 的值是 0
给你一个字符串数组 operations ，这是由操作组成的一个列表，返回执行所有操作后， X 的 最终值 。
 
示例 1：
输入：operations = ["--X","X++","X++"]
输出：1
解释：操作按下述步骤执行：
最初，X = 0
--X：X 减 1 ，X =  0 - 1 = -1
X++：X 加 1 ，X = -1 + 1 =  0
X++：X 加 1 ，X =  0 + 1 =  1
示例 2：
输入：operations = ["++X","++X","X++"]
输出：3
解释：操作按下述步骤执行： 
最初，X = 0
++X：X 加 1 ，X = 0 + 1 = 1
++X：X 加 1 ，X = 1 + 1 = 2
X++：X 加 1 ，X = 2 + 1 = 3
示例 3：
输入：operations = ["X++","++X","--X","X--"]
输出：0
解释：操作按下述步骤执行：
最初，X = 0
X++：X 加 1 ，X = 0 + 1 = 1
++X：X 加 1 ，X = 1 + 1 = 2
--X：X 减 1 ，X = 2 - 1 = 1
X--：X 减 1 ，X = 1 - 1 = 0
```

`思路`
遍历operations，有'+'答案+1，否则-1;

`c# 实现`
```
public class Solution {
    public int FinalValueAfterOperations(string[] operations) {
        int res = 0;
        for (int i = 0; i < operations.Length; i++)
        {
            int index = operations[i].IndexOf('+');
            if (index == -1)
            {
                res --;
            }else{
                res ++;
            }
        }
        return res;
    }
}
```

***

## 2022/12/21

## 1753. 移除石子的最大得分

[1753. 移除石子的最大得分](https://leetcode.cn/problems/maximum-score-from-removing-stones/submissions/390256552/)
```
你正在玩一个单人游戏，面前放置着大小分别为 a 、b 和 c 的 三堆 石子。
每回合你都要从两个 不同的非空堆 中取出一颗石子，并在得分上加 1 分。当存在 两个或更多 的空堆时，游戏停止。
给你三个整数 a 、b 和 c ，返回可以得到的 最大分数 。
 
示例 1：
输入：a = 2, b = 4, c = 6
输出：6
解释：石子起始状态是 (2, 4, 6) ，最优的一组操作是：
- 从第一和第三堆取，石子状态现在是 (1, 4, 5)
- 从第一和第三堆取，石子状态现在是 (0, 4, 4)
- 从第二和第三堆取，石子状态现在是 (0, 3, 3)
- 从第二和第三堆取，石子状态现在是 (0, 2, 2)
- 从第二和第三堆取，石子状态现在是 (0, 1, 1)
- 从第二和第三堆取，石子状态现在是 (0, 0, 0)
总分：6 分 。
示例 2：
输入：a = 4, b = 4, c = 6
输出：7
解释：石子起始状态是 (4, 4, 6) ，最优的一组操作是：
- 从第一和第二堆取，石子状态现在是 (3, 3, 6)
- 从第一和第三堆取，石子状态现在是 (2, 3, 5)
- 从第一和第三堆取，石子状态现在是 (1, 3, 4)
- 从第一和第三堆取，石子状态现在是 (0, 3, 3)
- 从第二和第三堆取，石子状态现在是 (0, 2, 2)
- 从第二和第三堆取，石子状态现在是 (0, 1, 1)
- 从第二和第三堆取，石子状态现在是 (0, 0, 0)
总分：7 分 。
示例 3：
输入：a = 1, b = 8, c = 8
输出：8
解释：最优的一组操作是连续从第二和第三堆取 8 回合，直到将它们取空。
注意，由于第二和第三堆已经空了，游戏结束，不能继续从第一堆中取石子。
```

`思路`
理想情况最大分数为 a b c的和除以2，如果 a b c 中较小的两个和比理想分数小，则答案应为a b c 中较小的两个和

`c# 实现`
```
public class Solution {
    public int MaximumScore(int a, int b, int c) {
        int min = Math.Min(a + b , a + c);
        min = Math.Min(min , b + c);
        return Math.Min((a + b + c)/2 , min);
    }
}
```

***

## 2022/12/20

## 1760. 袋子里最少数目的球

[1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/description/)
```
给你一个整数数组 nums ，其中 nums[i] 表示第 i 个袋子里球的数目。同时给你一个整数 maxOperations 。

你可以进行如下操作至多 maxOperations 次：

选择任意一个袋子，并将袋子里的球分到 2 个新的袋子中，每个袋子里都有 正整数 个球。
比方说，一个袋子里有 5 个球，你可以把它们分到两个新袋子里，分别有 1 个和 4 个球，或者分别有 2 个和 3 个球。
你的开销是单个袋子里球数目的 最大值 ，你想要 最小化 开销。

请你返回进行上述操作后的最小开销。

 

示例 1：

输入：nums = [9], maxOperations = 2
输出：3
解释：
- 将装有 9 个球的袋子分成装有 6 个和 3 个球的袋子。[9] -> [6,3] 。
- 将装有 6 个球的袋子分成装有 3 个和 3 个球的袋子。[6,3] -> [3,3,3] 。
装有最多球的袋子里装有 3 个球，所以开销为 3 并返回 3 。
示例 2：

输入：nums = [2,4,8,2], maxOperations = 4
输出：2
解释：
- 将装有 8 个球的袋子分成装有 4 个和 4 个球的袋子。[2,4,8,2] -> [2,4,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,4,4,4,2] -> [2,2,2,4,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,4,4,2] -> [2,2,2,2,2,4,2] 。
- 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,2,2,4,2] -> [2,2,2,2,2,2,2,2] 。
装有最多球的袋子里装有 2 个球，所以开销为 2 并返回 2 。
示例 3：

输入：nums = [7,17], maxOperations = 2
输出：7
```

`思路`
使用二分查找
假设最小开销为x，可以求得每个数需要的操作数，操作数总和只要小于maxOperations则记录当次的答案并缩小right，否则增大left

`c# 实现`
```
public class Solution {
    public int MinimumSize(int[] nums, int maxOperations) {
        int left = 1,right = nums.Max();
        int ans = 0;
        while(left <= right){
            int mid = (left + right)/2;
            int opCnt = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                opCnt += (nums[i] - 1)/mid;
            }
            if (opCnt <= maxOperations)
            {
                ans = mid;
                right = mid - 1;
            }else{
                left = mid + 1;
            }
        }
        return ans;
    }
}
```

***

## 2022/12/13

## 1832. 判断句子是否为全字母句

[1832. 判断句子是否为全字母句](https://leetcode.cn/problems/check-if-the-sentence-is-pangram/description/)
```
全字母句 指包含英语字母表中每个字母至少一次的句子。

给你一个仅由小写英文字母组成的字符串 sentence ，请你判断 sentence 是否为 全字母句 。

如果是，返回 true ；否则，返回 false 。

 

示例 1：

输入：sentence = "thequickbrownfoxjumpsoverthelazydog"
输出：true
解释：sentence 包含英语字母表中每个字母至少一次。
示例 2：

输入：sentence = "leetcode"
输出：false
```

`思路`
遍历用哈希表记录所有字符

`c# 实现`
```
public class Solution {
    public bool CheckIfPangram(string sentence) {
        if (sentence.Length < 26)
        {
            return false;
        }
        ISet<char> set = new HashSet<char>();
        for (int i = 0; i < sentence.Length; i++)
        {
            set.Add(sentence[i]);
        }
        return set.Count == 26;
    }
}
```

***

## 2022/12/12

## 1781. 所有子字符串美丽值之和

[1781. 所有子字符串美丽值之和](https://leetcode.cn/problems/sum-of-beauty-of-all-substrings/submissions/388584716/)
```
一个字符串的 美丽值 定义为：出现频率最高字符与出现频率最低字符的出现次数之差。

比方说，"abaacc" 的美丽值为 3 - 1 = 2 。
给你一个字符串 s ，请你返回它所有子字符串的 美丽值 之和。
```

`思路`
双重遍历+哈希表

`c# 实现`
```
public class Solution {
    public int BeautySum(string s) {
        int res = 0;
        for (int i = 0; i < s.Length - 2; i++)
        {
            int[] cnt = new int[26];
            int max = 0;
            for (int j = i; j < s.Length; j++)
            {
                cnt[s[j] - 'a']++;
                max = Math.Max(max,cnt[s[j] - 'a']);
                int min = s.Length;
                for (int k = 0; k < 26; k++)
                {
                    if (cnt[k] > 0)
                    {
                        min = Math.Min(min,cnt[k]);
                    }
                }
                res += max - min;
            }
        }
        return res;
    }
}

```

***

## 2022/12/9

✨## 1780. 判断一个数字是否可以表示成三的幂的和

[1780. 判断一个数字是否可以表示成三的幂的和](https://leetcode.cn/problems/check-if-number-is-a-sum-of-powers-of-three/description/)
```
给你一个整数 n ，如果你可以将 n 表示成若干个不同的三的幂之和，请你返回 true ，否则请返回 false 。

对于一个整数 y ，如果存在整数 x 满足 y == 3x ，我们称这个整数 y 是三的幂。

 

示例 1：

输入：n = 12
输出：true
解释：12 = 31 + 32
示例 2：

输入：n = 91
输出：true
解释：91 = 30 + 32 + 34
示例 3：

输入：n = 21
输出：false
```

`思路`
很巧妙的一道题，思路是将n转换成3进制来表达，比如12的3进制位 (110) 也就是12可以由 一个3的2次方 + 一个3的一次方 + 0 = 12 来构成，我们把n转换成3进制，每一位上只要是0或者1即可满足题目**不同**的三的幂之和的要求，如果有2则说明有重复的次幂则不满足要求，实现时我们不断对n除3并取余，除3的目的是使n右移（类比二进制的>>操作）来判断每一位上是否有2.

`c# 实现`
```
public class Solution {
    public bool CheckPowersOfThree(int n) {
        while(n != 0){
            if(n % 3 == 2){
                return false;
            }
            n /= 3;
        }
        return true;
    }
}
```

***

## 2022/12/7

## 1775. 通过最少操作次数使数组的和相等

[1775. 通过最少操作次数使数组的和相等](https://leetcode.cn/problems/equal-sum-arrays-with-minimum-number-of-operations/description/)
```
给你两个长度可能不等的整数数组 nums1 和 nums2 。两个数组中的所有值都在 1 到 6 之间（包含 1 和 6）。

每次操作中，你可以选择 任意 数组中的任意一个整数，将它变成 1 到 6 之间 任意 的值（包含 1 和 6）。

请你返回使 nums1 中所有数的和与 nums2 中所有数的和相等的最少操作次数。如果无法使两个数组的和相等，请返回 -1 。

 

示例 1：

输入：nums1 = [1,2,3,4,5,6], nums2 = [1,1,2,2,2,2]
输出：3
解释：你可以通过 3 次操作使 nums1 中所有数的和与 nums2 中所有数的和相等。以下数组下标都从 0 开始。
- 将 nums2[0] 变为 6 。 nums1 = [1,2,3,4,5,6], nums2 = [6,1,2,2,2,2] 。
- 将 nums1[5] 变为 1 。 nums1 = [1,2,3,4,5,1], nums2 = [6,1,2,2,2,2] 。
- 将 nums1[2] 变为 2 。 nums1 = [1,2,2,4,5,1], nums2 = [6,1,2,2,2,2] 。
示例 2：

输入：nums1 = [1,1,1,1,1,1,1], nums2 = [6]
输出：-1
解释：没有办法减少 nums1 的和或者增加 nums2 的和使二者相等。
示例 3：

输入：nums1 = [6,6], nums2 = [1]
输出：3
解释：你可以通过 3 次操作使 nums1 中所有数的和与 nums2 中所有数的和相等。以下数组下标都从 0 开始。
- 将 nums1[0] 变为 2 。 nums1 = [2,6], nums2 = [1] 。
- 将 nums1[1] 变为 2 。 nums1 = [2,2], nums2 = [1] 。
- 将 nums2[0] 变为 4 。 nums1 = [2,2], nums2 = [4] 。
```

`思路`
哈希表记录
要求num1 nums2和相同,先求出nums1 nums2 的差 diff，问题转化为修改两数组的元素，使diff归0， 两数组中每个数字位1-6，也就是每次修改元素最多能使diff减少5最小减少1，
排序数组后，遍历num1 nums2 , 记录使diff减少贡献1-5的个数，最后遍历哈希表，计算最少次数

`c# 实现`
```
public class Solution {
    public int MinOperations(int[] nums1, int[] nums2) {
        int res = 0;
        int sum1 = nums1.Sum();
        int sum2 = nums2.Sum();
        int diff = Math.Abs(sum1 - sum2);  
        int[] descNums , asceNums;
        if (sum1 > sum2)
        {
            Array.Sort(nums2,(a,b)=>{
                return a - b;
            });
            Array.Sort(nums1,(a,b)=>{
                return b - a;
            });
            descNums = nums1;
            asceNums = nums2;
        }else{
            Array.Sort(nums2,(a,b)=>{
                return b - a;
            });
            Array.Sort(nums1,(a,b)=>{
                return a - b;
            });
            descNums = nums2;
            asceNums = nums1;
        }
        // Dictionary<int,int> dict = new Dictionary<int,int>();
        int[] hash = new int[6];
        int n = asceNums.Length > descNums.Length ? asceNums.Length : descNums.Length;
        for (int i = 0; i < n; i++)
        {
            int sub = 0, add = 0;
            if (i < descNums.Length)
            {
                sub = descNums[i] - 1;
            }
            if (i < asceNums.Length)
            {
                add = 6 - asceNums[i];
            }
            if (sub > 0)
            {
                hash[sub] ++;
            }
            if (add > 0)
            {
                hash[add] ++;
            }
        }
        for (int i = 5; i > 0; i--)
        {
            int t = Math.Min((diff + i - 1) / i, hash[i]);
            res += t;
            diff -= t * i;
            if (diff <= 0)
            {
                return res;
            }
        }
        return -1;
    }
}
```

***

## 2022/12/6

## 1805. 字符串中不同整数的数目

[1805. 字符串中不同整数的数目](https://leetcode.cn/problems/number-of-different-integers-in-a-string/description/)
```
给你一个字符串 word ，该字符串由数字和小写英文字母组成。
请你用空格替换每个不是数字的字符。例如，"a123bc34d8ef34" 将会变成 " 123  34 8  34" 。注意，剩下的这些整数为（相邻彼此至少有一个空格隔开）："123"、"34"、"8" 和 "34" 。
返回对 word 完成替换后形成的 不同 整数的数目。
只有当两个整数的 不含前导零 的十进制表示不同， 才认为这两个整数也不同。
 
示例 1：
输入：word = "a123bc34d8ef34"
输出：3
解释：不同的整数有 "123"、"34" 和 "8" 。注意，"34" 只计数一次。
示例 2：
输入：word = "leet1234code234"
输出：2
示例 3：
输入：word = "a1b01c001"
输出：1
解释："1"、"01" 和 "001" 视为同一个整数的十进制表示，因为在比较十进制值时会忽略前导零的存在。

```

`思路`
双指针遍历字符串

`c# 实现`
```
public class Solution {
    public int NumDifferentIntegers(string word) {
        ISet<string> set = new HashSet<string>();
        int n = word.Length, left = 0, right;
        while(true){
            while(left < n && !char.IsDigit(word[left])){
                left++;
            }
            if (left == n)
            {
                break;
            }
            right = left + 1;
            while(right < n && char.IsDigit(word[right])){
                right ++;
            }
            while(right - left > 1 && word[left] == '0'){
                left ++;
            }
            set.Add(word.Substring(left,right - left));
            left = right;
        }
        return set.Count;
    }
}
```

***

## 2022/12/2

## 1769. 移动所有球到每个盒子所需的最小操作数

[1769. 移动所有球到每个盒子所需的最小操作数](https://leetcode.cn/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/description/)
```
有 n 个盒子。给你一个长度为 n 的二进制字符串 boxes ，其中 boxes[i] 的值为 '0' 表示第 i 个盒子是 空 的，而 boxes[i] 的值为 '1' 表示盒子里有 一个 小球。
在一步操作中，你可以将 一个 小球从某个盒子移动到一个与之相邻的盒子中。第 i 个盒子和第 j 个盒子相邻需满足 abs(i - j) == 1 。注意，操作执行后，某些盒子中可能会存在不止一个小球。
返回一个长度为 n 的数组 answer ，其中 answer[i] 是将所有小球移动到第 i 个盒子所需的 最小 操作数。
每个 answer[i] 都需要根据盒子的 初始状态 进行计算。
 
示例 1：
输入：boxes = "110"
输出：[1,1,3]
解释：每个盒子对应的最小操作数如下：
1) 第 1 个盒子：将一个小球从第 2 个盒子移动到第 1 个盒子，需要 1 步操作。
2) 第 2 个盒子：将一个小球从第 1 个盒子移动到第 2 个盒子，需要 1 步操作。
3) 第 3 个盒子：将一个小球从第 1 个盒子移动到第 3 个盒子，需要 2 步操作。将一个小球从第 2 个盒子移动到第 3 个盒子，需要 1 步操作。共计 3 步操作。
示例 2：
输入：boxes = "001011"
输出：[11,8,5,4,3,4]

```

`思路`
直接模拟时间为O(n平方)，可以遍历两次，对于当前i位置，操作次数sum，如果到i+1,sum为上一次sum + （i +1)左边的1个个数 - (i+1)右边1的个数

`c# 实现`
```
public class Solution {
    public int[] MinOperations(string boxes) {
        int left = boxes[0] - '0', right = 0, sum = 0;
        for (int i = 1; i < boxes.Length; i++)
        {
            if (boxes[i] == '1')
            {
                sum += i;
                right ++;
            }
        }
        int[] ans = new int[boxes.Length];
        ans[0] = sum;
        for (int i = 1; i < boxes.Length; i++)
        {
            sum +=  left - right;
            if (boxes[i] == '1'){
                left ++;
                right --;
            }
            ans[i] = sum;
        }
        return ans;
    }
}
```

***

## 2022/12/1

## 1779. 找到最近的有相同 X 或 Y 坐标的点

[1779. 找到最近的有相同 X 或 Y 坐标的点](https://leetcode.cn/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/description/)
```
给你两个整数 x 和 y ，表示你在一个笛卡尔坐标系下的 (x, y) 处。同时，在同一个坐标系下给你一个数组 points ，其中 points[i] = [ai, bi] 表示在 (ai, bi) 处有一个点。当一个点与你所在的位置有相同的 x 坐标或者相同的 y 坐标时，我们称这个点是 有效的 。
请返回距离你当前位置 曼哈顿距离 最近的 有效 点的下标（下标从 0 开始）。如果有多个最近的有效点，请返回下标 最小 的一个。如果没有有效点，请返回 -1 。
两个点 (x1, y1) 和 (x2, y2) 之间的 曼哈顿距离 为 abs(x1 - x2) + abs(y1 - y2) 。
 
示例 1：
输入：x = 3, y = 4, points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
输出：2
解释：所有点中，[3,1]，[2,4] 和 [4,4] 是有效点。有效点中，[2,4] 和 [4,4] 距离你当前位置的曼哈顿距离最小，都为 1 。[2,4] 的下标最小，所以返回 2 。
示例 2：
输入：x = 3, y = 4, points = [[3,4]]
输出：0
提示：答案可以与你当前所在位置坐标相同。
示例 3：
输入：x = 3, y = 4, points = [[2,3]]
输出：-1
解释：没有 有效点。

```

`思路`
遍历数组，根据要求过滤不符合的点

`c# 实现`
```
public class Solution {
    public int NearestValidPoint(int x, int y, int[][] points) {
        int ans = -1 , distance = int.MaxValue;
        for (int i = 0; i < points.Length; i++)
        {
            int px = points[i][0];
            int py = points[i][1];
            if (px == x || py == y)
            {
                int disx = Math.Abs(points[i][0] - x) +  Math.Abs(points[i][1] - y);
                if (disx <  distance)
                {
                    distance = disx;
                    ans = i;
                }
            }
         
        }
        return ans;
    }
}
```

***

## 2022/11/30

## 895. 最大频率栈

[895. 最大频率栈](https://leetcode.cn/problems/maximum-frequency-stack/description/)
```
设计一个类似堆栈的数据结构，将元素推入堆栈，并从堆栈中弹出出现频率最高的元素。

实现 FreqStack 类:

FreqStack() 构造一个空的堆栈。
void push(int val) 将一个整数 val 压入栈顶。
int pop() 删除并返回堆栈中出现频率最高的元素。
如果出现频率最高的元素不只一个，则移除并返回最接近栈顶的元素。
 

示例 1：

输入：
["FreqStack","push","push","push","push","push","push","pop","pop","pop","pop"],
[[],[5],[7],[5],[7],[4],[5],[],[],[],[]]
输出：[null,null,null,null,null,null,null,5,7,5,4]
解释：
FreqStack = new FreqStack();
freqStack.push (5);//堆栈为 [5]
freqStack.push (7);//堆栈是 [5,7]
freqStack.push (5);//堆栈是 [5,7,5]
freqStack.push (7);//堆栈是 [5,7,5,7]
freqStack.push (4);//堆栈是 [5,7,5,7,4]
freqStack.push (5);//堆栈是 [5,7,5,7,4,5]
freqStack.pop ();//返回 5 ，因为 5 出现频率最高。堆栈变成 [5,7,5,7,4]。
freqStack.pop ();//返回 7 ，因为 5 和 7 出现频率最高，但7最接近顶部。堆栈变成 [5,7,5,4]。
freqStack.pop ();//返回 5 ，因为 5 出现频率最高。堆栈变成 [5,7,4]。
freqStack.pop ();//返回 4 ，因为 4, 5 和 7 出现频率最高，但 4 是最接近顶部的。堆栈变成 [5,7]。
 

提示：

0 <= val <= 109
push 和 pop 的操作数不大于 2 * 104。
输入保证在调用 pop 之前堆栈中至少有一个元素。
```

`思路`
用两个字典分别记录单个数字的出现次数和每个次数的对应数字的栈

`c# 实现`
```
public class FreqStack {
    private IDictionary<int, int> freq;
    private IDictionary<int, Stack<int>> group;
    private int maxFreq;

    public FreqStack() {
        freq = new Dictionary<int, int>();
        group = new Dictionary<int, Stack<int>>();
        maxFreq = 0;
    }
    
    public void Push(int val) {
        if (!freq.ContainsKey(val))
        {
            freq.Add(val,0);
        }
        freq[val] ++;
        if (!group.ContainsKey(freq[val]))
        {
            group.Add(freq[val] , new Stack<int>());
        }
        group[freq[val]].Push(val);
        maxFreq = Math.Max(maxFreq, freq[val]);
    }
    
    public int Pop() {
        int res = 0;
        if (group.ContainsKey(maxFreq))
        {
            res = group[maxFreq].Pop();
            freq[res] --;
            if (group[maxFreq].Count == 0)
            {
                maxFreq--;
            }
        }

        return res;
    }
}
```

***

## 2022/11/29

## 1758. 生成交替二进制字符串的最少操作数

[1758. 生成交替二进制字符串的最少操作数](https://leetcode.cn/problems/minimum-changes-to-make-alternating-binary-string/description/)
```

给你一个仅由字符 '0' 和 '1' 组成的字符串 s 。一步操作中，你可以将任一 '0' 变成 '1' ，或者将 '1' 变成 '0' 。

交替字符串 定义为：如果字符串中不存在相邻两个字符相等的情况，那么该字符串就是交替字符串。例如，字符串 "010" 是交替字符串，而字符串 "0100" 不是。

返回使 s 变成 交替字符串 所需的 最少 操作数。

 

示例 1：

输入：s = "0100"
输出：1
解释：如果将最后一个字符变为 '1' ，s 就变成 "0101" ，即符合交替字符串定义。
示例 2：

输入：s = "10"
输出：0
解释：s 已经是交替字符串。
示例 3：

输入：s = "1111"
输出：2
解释：需要 2 步操作得到 "0101" 或 "1010" 。
```

`思路`
遍历s,分别记录0开头和1开头情况下，s字符串不符合交替的个数

`c# 实现`
```
public class Solution {
    public int MinOperations(string s) {
        int ans1 = 0 , ans2 = 0;
        for (int i = 0; i < s.Length; i++)
        {
            int num = s[i] - '0';
            if ((i & 1) == 1)
            {
                if (num == 0)
                {
                    ans1 ++;
                }else{
                    ans2 ++;
                }
            }else{
                if (num != 0)
                {
                    ans1 ++;
                }else{
                    ans2 ++;
                }
            }
        }
        return Math.Min(ans1,ans2);
    }
}

```

***

## 2022/11/25

## 809. 情感丰富的文字

[809. 情感丰富的文字](https://leetcode.cn/problems/expressive-words/description/)
```
809. 情感丰富的文字

有时候人们会用重复写一些字母来表示额外的感受，比如 "hello" -> "heeellooo", "hi" -> "hiii"。我们将相邻字母都相同的一串字符定义为相同字母组，例如："h", "eee", "ll", "ooo"。

对于一个给定的字符串 S ，如果另一个单词能够通过将一些字母组扩张从而使其和 S 相同，我们将这个单词定义为可扩张的（stretchy）。扩张操作定义如下：选择一个字母组（包含字母 c ），然后往其中添加相同的字母 c 使其长度达到 3 或以上。

例如，以 "hello" 为例，我们可以对字母组 "o" 扩张得到 "hellooo"，但是无法以同样的方法得到 "helloo" 因为字母组 "oo" 长度小于 3。此外，我们可以进行另一种扩张 "ll" -> "lllll" 以获得 "helllllooo"。如果 s = "helllllooo"，那么查询词 "hello" 是可扩张的，因为可以对它执行这两种扩张操作使得 query = "hello" -> "hellooo" -> "helllllooo" = s。

输入一组查询单词，输出其中可扩张的单词数量。

 

示例：

输入： 
s = "heeellooo"
words = ["hello", "hi", "helo"]
输出：1
解释：
我们能通过扩张 "hello" 的 "e" 和 "o" 来得到 "heeellooo"。
我们不能通过扩张 "helo" 来得到 "heeellooo" 因为 "ll" 的长度小于 3 。
```

`思路`
双指针寻找规律

`c# 实现`
```
public class Solution {
    public int ExpressiveWords(string s, string[] words) {
        int ans = 0;
        foreach (string word in words) {
            if (Expand(s, word)) {
                ++ans;
            }
        }
        return ans;
    }

    private bool Expand(string s, string t) {
        int i = 0, j = 0;
        while (i < s.Length && j < t.Length) {
            if (s[i] != t[j]) {
                return false;
            }
            char ch = s[i];
            int cnti = 0;
            while (i < s.Length && s[i] == ch) {
                ++cnti;
                ++i;
            }
            int cntj = 0;
            while (j < t.Length && t[j] == ch) {
                ++cntj;
                ++j;
            }
            if (cnti < cntj) {
                return false;
            }
            if (cnti != cntj && cnti < 3) {
                return false;
            }
        }
        return i == s.Length && j == t.Length;
    }
}

```

***

## 2022/11/24

## 795. 区间子数组个数

[795. 区间子数组个数](https://leetcode.cn/problems/number-of-subarrays-with-bounded-maximum/solutions/)
```
给你一个整数数组 nums 和两个整数：left 及 right 。找出 nums 中连续、非空且其中最大元素在范围 [left, right] 内的子数组，并返回满足条件的子数组的个数。
生成的测试用例保证结果符合 32-bit 整数范围。
 
示例 1：
输入：nums = [2,1,4,3], left = 2, right = 3
输出：3
解释：满足条件的三个子数组：[2], [2, 1], [3]
示例 2：
输入：nums = [2,9,2,5,6], left = 2, right = 8
输出：7
```

`思路`
分区计数

`c# 实现`
```
public class Solution {
    public int NumSubarrayBoundedMax(int[] nums, int left, int right) {
        int res = 0, last2 = -1, last1 = -1;
        for (int i = 0; i < nums.Length; i++) {
            if (nums[i] >= left && nums[i] <= right) {
                last1 = i;
            } else if (nums[i] > right) {
                last2 = i;
                last1 = -1;
            }
            if (last1 != -1) {
                res += last1 - last2;
            }
        }
        return res;
    }
}
```

***

## 2022/11/23

## 1742. 盒子中小球的最大数量

[1742. 盒子中小球的最大数量](https://leetcode.cn/problems/maximum-number-of-balls-in-a-box/description/)
```
你在一家生产小球的玩具厂工作，有 n 个小球，编号从 lowLimit 开始，到 highLimit 结束（包括 lowLimit 和 highLimit ，即 n == highLimit - lowLimit + 1）。另有无限数量的盒子，编号从 1 到 infinity 。

你的工作是将每个小球放入盒子中，其中盒子的编号应当等于小球编号上每位数字的和。例如，编号 321 的小球应当放入编号 3 + 2 + 1 = 6 的盒子，而编号 10 的小球应当放入编号 1 + 0 = 1 的盒子。

给你两个整数 lowLimit 和 highLimit ，返回放有最多小球的盒子中的小球数量。如果有多个盒子都满足放有最多小球，只需返回其中任一盒子的小球数量。

 

示例 1：

输入：lowLimit = 1, highLimit = 10
输出：2
解释：
盒子编号：1 2 3 4 5 6 7 8 9 10 11 ...
小球数量：2 1 1 1 1 1 1 1 1 0  0  ...
编号 1 的盒子放有最多小球，小球数量为 2 。
示例 2：

输入：lowLimit = 5, highLimit = 15
输出：2
解释：
盒子编号：1 2 3 4 5 6 7 8 9 10 11 ...
小球数量：1 1 1 1 2 2 1 1 1 0  0  ...
编号 5 和 6 的盒子放有最多小球，每个盒子中的小球数量都是 2 。
示例 3：

输入：lowLimit = 19, highLimit = 28
输出：2
解释：
盒子编号：1 2 3 4 5 6 7 8 9 10 11 12 ...
小球数量：0 1 1 1 1 1 1 1 1 2  0  0  ...
编号 10 的盒子放有最多小球，小球数量为 2 。
```

`思路`
遍历所有数，计数位数相加最多的一个

`c# 实现`
```
public class Solution {
    public int CountBalls(int lowLimit, int highLimit) {
        int max = 0;
        Dictionary<int , int> dict = new Dictionary<int , int>();
        for (int i = lowLimit; i <= highLimit; i++)
        {
            int num = i;
            int order = 0;
            while (num != 0)
            {
                order += num % 10;
                num /= 10;
            }
            dict.TryAdd(order,0);
            dict[order]++;
            max = Math.Max(max, dict[order]);
        }
        return max;
    }
}
```

***



## 2022/11/22

## ✨878. 第 N 个神奇数字

[878. 第 N 个神奇数字](https://leetcode.cn/problems/nth-magical-number/description/)
```
一个正整数如果能被 a 或 b 整除，那么它是神奇的。

给定三个整数 n , a , b ，返回第 n 个神奇的数字。因为答案可能很大，所以返回答案 对 109 + 7 取模 后的值。

 

示例 1：

输入：n = 1, a = 2, b = 3
输出：2
示例 2：

输入：n = 4, a = 2, b = 3
输出：6
```

`思路`
直接模拟会超时，主要思路是使用二分查找来优化，题目要求第 n 个神奇的数字，假设 a 和 b 中较小的一个是 a ，那么结果的范围就是[a , n * a],从而变成了在[a , n * a]中找到第 n 个神奇的数字，其中n满足被 a 整除的个数 + 被 b 整除的个数 - 被 ab 整除的个数。使用二分查找求出最后的答案。

`c# 实现`
```
public class Solution {
    const int MOD = 1000000007;
    public int NthMagicalNumber(int n, int a, int b) {
        long l = Math.Min(a,b);
        long r = l * n;
        int c = LCM(a,b);
        while(l <= r){
            long mid = (r + l)/2;
            long cnt = mid/a + mid/b - mid/c;
            if (cnt >= n)
            {
                r = mid - 1;
            }else{
                l = mid + 1;
            }
        }
        return (int)((r+1)%MOD);
    }

    private int LCM(int a,int b){
        return a*b/GCD(a,b);
    }

    private int GCD(int a,int b){
        return b != 0 ? GCD(b,a%b) : a;
    }
}
```

***

## 2022/11/21

## ✨808. 分汤

[808. 分汤](https://leetcode.cn/problems/soup-servings/description/)
```
有 A 和 B 两种类型 的汤。一开始每种类型的汤有 n 毫升。有四种分配操作：

提供 100ml 的 汤A 和 0ml 的 汤B 。
提供 75ml 的 汤A 和 25ml 的 汤B 。
提供 50ml 的 汤A 和 50ml 的 汤B 。
提供 25ml 的 汤A 和 75ml 的 汤B 。
当我们把汤分配给某人之后，汤就没有了。每个回合，我们将从四种概率同为 0.25 的操作中进行分配选择。如果汤的剩余量不足以完成某次操作，我们将尽可能分配。当两种类型的汤都分配完时，停止操作。

注意 不存在先分配 100 ml 汤B 的操作。

需要返回的值： 汤A 先分配完的概率 +  汤A和汤B 同时分配完的概率 / 2。返回值在正确答案 10-5 的范围内将被认为是正确的。

 

示例 1:

输入: n = 50
输出: 0.62500
解释:如果我们选择前两个操作，A 首先将变为空。
对于第三个操作，A 和 B 会同时变为空。
对于第四个操作，B 首先将变为空。
所以 A 变为空的总概率加上 A 和 B 同时变为空的概率的一半是 0.25 *(1 + 1 + 0.5 + 0)= 0.625。
示例 2:

输入: n = 100
输出: 0.71875
```

`思路`
见[官方题解](https://leetcode.cn/problems/soup-servings/solutions/1981704/fen-tang-by-leetcode-solution-0yxs/)

`c# 实现`
```
public class Solution {
    public double SoupServings(int n) {
        n = (int) Math.Ceiling((double) n / 25);
        if (n >= 179) {
            return 1.0;
        }
        double[][] dp = new double[n + 1][];
        for (int i = 0; i <= n; i++) {
            dp[i] = new double[n + 1];
        }
        dp[0][0] = 0.5;
        for (int i = 1; i <= n; i++) {
            dp[0][i] = 1.0;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (dp[Math.Max(0, i - 4)][j] + dp[Math.Max(0, i - 3)][Math.Max(0, j - 1)] + dp[Math.Max(0, i - 2)][Math.Max(0, j - 2)] + dp[Math.Max(0, i - 1)][Math.Max(0, j - 3)]) / 4.0;
            }
        }
        return dp[n][n];
    }
}
```

***

## 2022/11/18

## 891. 子序列宽度之和

[891. 子序列宽度之和](https://leetcode.cn/problems/sum-of-subsequence-widths/description/)
```
一个序列的 宽度 定义为该序列中最大元素和最小元素的差值。

给你一个整数数组 nums ，返回 nums 的所有非空 子序列 的 宽度之和 。由于答案可能非常大，请返回对 109 + 7 取余 后的结果。

子序列 定义为从一个数组里删除一些（或者不删除）元素，但不改变剩下元素的顺序得到的数组。例如，[3,6,2,7] 就是数组 [0,3,1,6,2,2,7] 的一个子序列。

 

示例 1：

输入：nums = [2,1,3]
输出：6
解释：子序列为 [1], [2], [3], [2,1], [2,3], [1,3], [2,1,3] 。
相应的宽度是 0, 0, 0, 1, 1, 2, 2 。
宽度之和是 6 。
示例 2：

输入：nums = [2]
输出：0
```

`思路`
O(n平方)解法
先对nums排序，遍历nums，计算每个以i开始，j结束的区间的子序列个数，累加这些值，需要注意的是，因为Math.Pow()返回的是double，计算过程中会超过double最大值，导致结果不正确，所有不能使用Math.Pow()方法。

O(n)的解法
参考leetcode一位用户的[题解](https://leetcode.cn/problems/sum-of-subsequence-widths/solutions/1977772/by-muse-77-f6s5/)

`c# 实现`
```
1.O(n平方)解法
public class Solution {
    public int SumSubseqWidths(int[] nums) {
        const int MOD = 1000000007;
        Array.Sort(nums);
        long sum = 0;
        long[] pow = new long[nums.Length];
        pow[0] = 1;
        for (int i = 1; i < nums.Length; i++) 
            pow[i] = (pow[i - 1] << 1) % MOD; // 初始化2^n的值
        for (int i = 0; i < nums.Length; i++)
        {
            for (int j = i + 1; j < nums.Length; j++)
            {
                long cnt = pow[j - i - 1];
                sum = (sum + ((nums[j] - nums[i])*cnt)%MOD)%MOD;
            }
        }
        return (int) ((sum + MOD) % MOD);
    }
}

2.O(n)的解法
public class Solution {
    public int SumSubseqWidths(int[] nums) {
        const int MOD = 1000000007;
        Array.Sort(nums);
        long ans = 0;
        int n = nums.Length;
        long[] pow = new long[n];
        pow[0] = 1;
        for (int i = 1; i < n; i++) 
            pow[i] = (pow[i - 1] << 1) % MOD; // 初始化2^n的值

        for (int i = 0; i < n; i++)
        {
            ans = (ans + (pow[i] - pow[n - i - 1]) * nums[i] % MOD) % MOD; 
            // ans = (ans + (( - )*nums[i])%MOD)%MOD;
        }
        return (int) ans;
    }
}
```

***

## 2022/11/17

## 792. 匹配子序列的单词数

[792. 匹配子序列的单词数](https://leetcode.cn/problems/number-of-matching-subsequences/description/)
```
给定字符串 s 和字符串数组 words, 返回  words[i] 中是s的子序列的单词个数 。

字符串的 子序列 是从原始字符串中生成的新字符串，可以从中删去一些字符(可以是none)，而不改变其余字符的相对顺序。

例如， “ace” 是 “abcde” 的子序列。
 

示例 1:

输入: s = "abcde", words = ["a","bb","acd","ace"]
输出: 3
解释: 有三个是 s 的子序列的单词: "a", "acd", "ace"。
Example 2:

输入: s = "dsahjpjauf", words = ["ahjpjau","ja","ahbwzgqnuk","tnmlanowax"]
输出: 2
```

`思路`
对于每个word的字符char,遍历s 当s[i]与char相等，寻找word下一个char，如果所有char都在s中出现，则ans+1

`c# 实现`
```
public class Solution {
    public int NumMatchingSubseq(string s, string[] words) {
        int ans = 0;
        foreach (string word in words)
        {
            int index = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == word[index])
                {
                    index++;
                    if (index >= word.Length)
                    {
                        ans++;
                        break;
                    }
                }
            }
          
        }
        return ans;
    }
}
```

***

## 2022/11/16

## 775. 全局倒置与局部倒置

[775. 全局倒置与局部倒置](https://leetcode.cn/problems/global-and-local-inversions/description/)
```
给你一个长度为 n 的整数数组 nums ，表示由范围 [0, n - 1] 内所有整数组成的一个排列。

全局倒置 的数目等于满足下述条件不同下标对 (i, j) 的数目：

0 <= i < j < n
nums[i] > nums[j]
局部倒置 的数目等于满足下述条件的下标 i 的数目：

0 <= i < n - 1
nums[i] > nums[i + 1]
当数组 nums 中 全局倒置 的数量等于 局部倒置 的数量时，返回 true ；否则，返回 false 。

 

示例 1：

输入：nums = [1,0,2]
输出：true
解释：有 1 个全局倒置，和 1 个局部倒置。
示例 2：

输入：nums = [1,2,0]
输出：false
解释：有 2 个全局倒置，和 1 个局部倒置。
```

`思路`
从后往前遍历nums，记录最小minSuff，只要有比过程中最小值还大的元素，则不满足

`c# 实现`
```
public class Solution {
    public bool IsIdealPermutation(int[] nums) {
        int n = nums.Length, minSuff = nums[n - 1];
        for (int i = n - 3; i >= 0; i--) {
            if (nums[i] > minSuff) {
                return false;
            }
            minSuff = Math.Min(minSuff, nums[i + 1]);
        }
        return true;
    }
}

```

***

## 2022/11/15

## 1710. 卡车上的最大单元数

[1710. 卡车上的最大单元数]()
```
请你将一些箱子装在 一辆卡车 上。给你一个二维数组 boxTypes ，其中 boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi] ：

numberOfBoxesi 是类型 i 的箱子的数量。
numberOfUnitsPerBoxi 是类型 i 每个箱子可以装载的单元数量。
整数 truckSize 表示卡车上可以装载 箱子 的 最大数量 。只要箱子数量不超过 truckSize ，你就可以选择任意箱子装到卡车上。

返回卡车可以装载 单元 的 最大 总数。

 

示例 1：

输入：boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
输出：8
解释：箱子的情况如下：
- 1 个第一类的箱子，里面含 3 个单元。
- 2 个第二类的箱子，每个里面含 2 个单元。
- 3 个第三类的箱子，每个里面含 1 个单元。
可以选择第一类和第二类的所有箱子，以及第三类的一个箱子。
单元总数 = (1 * 3) + (2 * 2) + (1 * 1) = 8
示例 2：

输入：boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
输出：91
```

`思路`
对boxTypes降序排序，遍历boxTypes，每次自加当前boxTypes的箱子的数量到答案ans，直到箱子大于truckSize

`c# 实现`
```
public class Solution {
    public int MaximumUnits(int[][] boxTypes, int truckSize) {
        Array.Sort(boxTypes,(int[] a, int[] b)=>{
            return b[1] - a[1];
        });
        int ans = 0;
        for (int i = 0; i < boxTypes.Length; i++)
        {
            int cnt = boxTypes[i][0] > truckSize ? truckSize : boxTypes[i][0];
            ans += boxTypes[i][1] * cnt;
            truckSize -= cnt;
            if (truckSize == 0)
            {
                return ans;
            }
        }
        return ans;
    }
}
```

***

## 2022/11/11

## 1704. 判断字符串的两半是否相似

[1704. 判断字符串的两半是否相似](https://leetcode.cn/problems/determine-if-string-halves-are-alike/description/)
```
给你一个偶数长度的字符串 s 。将其拆分成长度相同的两半，前一半为 a ，后一半为 b 。

两个字符串 相似 的前提是它们都含有相同数目的元音（'a'，'e'，'i'，'o'，'u'，'A'，'E'，'I'，'O'，'U'）。注意，s 可能同时含有大写和小写字母。

如果 a 和 b 相似，返回 true ；否则，返回 false 。

 

示例 1：

输入：s = "book"
输出：true
解释：a = "bo" 且 b = "ok" 。a 中有 1 个元音，b 也有 1 个元音。所以，a 和 b 相似。
示例 2：

输入：s = "textbook"
输出：false
解释：a = "text" 且 b = "book" 。a 中有 1 个元音，b 中有 2 个元音。因此，a 和 b 不相似。
注意，元音 o 在 b 中出现两次，记为 2 个。

```

`思路`
用hashset记录所有元音字符，遍历s,分别计数前半段和后半段的元音个数，返回两个计数是否相等。

`c# 实现`
```
public class Solution {
    public bool HalvesAreAlike(string s) {
        ISet<char> dict = new HashSet<char>();
        dict.Add('a');
        dict.Add('e');
        dict.Add('i');
        dict.Add('o');
        dict.Add('u');
        int half = s.Length/2;
        int left = 0 , right = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (dict.Contains(Char.ToLower(s[i])))
            {
                if (i < half)
                {
                    left++;
                }else{
                    right++;
                }
            }
        }
        return left == right;
    }
}
```

***

## 2022/11/

## 764. 最大加号标志

[764. 最大加号标志](https://leetcode.cn/problems/largest-plus-sign/description/)
```

中等
165
相关企业
在一个 n x n 的矩阵 grid 中，除了在数组 mines 中给出的元素为 0，其他每个元素都为 1。mines[i] = [xi, yi]表示 grid[xi][yi] == 0

返回  grid 中包含 1 的最大的 轴对齐 加号标志的阶数 。如果未找到加号标志，则返回 0 。

一个 k 阶由 1 组成的 “轴对称”加号标志 具有中心网格 grid[r][c] == 1 ，以及4个从中心向上、向下、向左、向右延伸，长度为 k-1，由 1 组成的臂。注意，只有加号标志的所有网格要求为 1 ，别的网格可能为 0 也可能为 1 。

 

示例 1：



输入: n = 5, mines = [[4, 2]]
输出: 2
解释: 在上面的网格中，最大加号标志的阶只能是2。一个标志已在图中标出。
示例 2：



输入: n = 1, mines = [[0, 0]]
输出: 0
解释: 没有加号标志，返回 0 。
```

`思路`
遍历grid,某个点的可能最大阶乘就是它前面连续有多少个1的计数，记录每个点上下左右可能阶乘的最小值，最后矩阵中最大的一个阶乘数就是答案

`c# 实现`
```
public class Solution {
    public int OrderOfLargestPlusSign(int n, int[][] mines) {
        int[][] dp = new int[n][];
        for (int i = 0; i < n; i++) {
            dp[i] = new int[n];
            Array.Fill(dp[i], n);
        }
        ISet<int> banned = new HashSet<int>();
        foreach (int[] vec in mines) {
            banned.Add(vec[0] * n + vec[1]);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int count = 0;
            /* left */
            for (int j = 0; j < n; j++) {
                if (banned.Contains(i * n + j)) {
                    count = 0;
                } else {
                    count++;
                }
                dp[i][j] = Math.Min(dp[i][j], count);
            }
            count = 0;
            /* right */ 
            for (int j = n - 1; j >= 0; j--) {
                if (banned.Contains(i * n + j)) {
                    count = 0;
                } else {
                    count++;
                }
                dp[i][j] = Math.Min(dp[i][j], count);
            }
        }
        for (int i = 0; i < n; i++) {
            int count = 0;
            /* up */
            for (int j = 0; j < n; j++) {
                if (banned.Contains(j * n + i)) {
                    count = 0;
                } else {
                    count++;
                }
                dp[j][i] = Math.Min(dp[j][i], count);
            }
            count = 0;
            /* down */
            for (int j = n - 1; j >= 0; j--) {
                if (banned.Contains(j * n + i)) {
                    count = 0;
                } else {
                    count++;
                }
                dp[j][i] = Math.Min(dp[j][i], count);
                ans = Math.Max(ans, dp[j][i]);
            }
        }
        return ans;
    }
}
```

***


## 2022/11/7

## 816. 模糊坐标

[816. 模糊坐标](https://leetcode.cn/problems/ambiguous-coordinates/description/)
```
816. 模糊坐标

我们有一些二维坐标，如 "(1, 3)" 或 "(2, 0.5)"，然后我们移除所有逗号，小数点和空格，得到一个字符串S。返回所有可能的原始字符串到一个列表中。

原始的坐标表示法不会存在多余的零，所以不会出现类似于"00", "0.0", "0.00", "1.0", "001", "00.01"或一些其他更小的数来表示坐标。此外，一个小数点前至少存在一个数，所以也不会出现“.1”形式的数字。

最后返回的列表可以是任意顺序的。而且注意返回的两个数字中间（逗号之后）都有一个空格。

 

示例 1:
输入: "(123)"
输出: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
示例 2:
输入: "(00011)"
输出:  ["(0.001, 1)", "(0, 0.011)"]
解释: 
0.0, 00, 0001 或 00.01 是不被允许的。
示例 3:
输入: "(0123)"
输出: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]
示例 4:
输入: "(100)"
输出: [(10, 0)]
解释: 
1.0 是不被允许的。
```

`思路`
去掉外层的括号，遍历s 将其分成左右两部分枚举所有合法的数字，最后拼接所有合法字符串。

`c# 实现`
```
public class Solution {
    public IList<string> AmbiguousCoordinates(string s) {
        int n = s.Length - 2;
        IList<string> res = new List<string>();
        s = s.Substring(1, s.Length - 2);
        for (int l = 1; l < n; ++l) {
            IList<string> lt = GetPos(s.Substring(0, l));
            if (lt.Count == 0) {
                continue;
            }
            IList<string> rt = GetPos(s.Substring(l));
            if (rt.Count == 0) {
                continue;
            }
            foreach (string i in lt) {
                foreach (string j in rt) {
                    res.Add("(" + i + ", " + j + ")");
                }
            }
        }
        return res;
    }

    private IList<string> GetPos(string s) {
        IList<string> pos = new List<string>();
        if (s[0] != '0' || "0".Equals(s))
        {
            pos.Add(s);
        }
        for (int i = 1; i < s.Length; i++)
        {
            if ((s[0] == '0' && i != 1) || s[s.Length - 1] == '0')
            {
                continue;
            }
            pos.Add(s.Substring(0, i) + "." + s.Substring(i));
        }
        return pos;
    }
}
```

***

## 2022/11/4

## 754. 到达终点数字

[754. 到达终点数字](https://leetcode.cn/problems/reach-a-number/description/)
```

在一根无限长的数轴上，你站在0的位置。终点在target的位置。

你可以做一些数量的移动 numMoves :

每次你可以选择向左或向右移动。
第 i 次移动（从  i == 1 开始，到 i == numMoves ），在选择的方向上走 i 步。
给定整数 target ，返回 到达目标所需的 最小 移动次数(即最小 numMoves ) 。

 

示例 1:

输入: target = 2
输出: 3
解释:
第一次移动，从 0 到 1 。
第二次移动，从 1 到 -1 。
第三次移动，从 -1 到 2 。
示例 2:

输入: target = 3
输出: 2
解释:
第一次移动，从 0 到 1 。
第二次移动，从 1 到 3 。
```

`思路`
计算1到k的和sum sum>=target 如果（sum - target）为偶数，则k为答案，否则k+1或者k+2中一定有一个为答案

`c# 实现`
```
public class Solution {
    public int ReachNumber(int target) {
        target = Math.Abs(target);
        int k = 0;
        while(target > 0){
            target -= ++k;
        }
        return (target & 1) == 0 ? k : k+1 + (k & 1);
    }
}
```

***



## 2022/11/2

## 1620. 网络信号最好的坐标

[1620. 网络信号最好的坐标](https://leetcode.cn/problems/coordinate-with-maximum-network-quality/)
```

给你一个数组 towers 和一个整数 radius 。

数组  towers  中包含一些网络信号塔，其中 towers[i] = [xi, yi, qi] 表示第 i 个网络信号塔的坐标是 (xi, yi) 且信号强度参数为 qi 。所有坐标都是在  X-Y 坐标系内的 整数 坐标。两个坐标之间的距离用 欧几里得距离 计算。

整数 radius 表示一个塔 能到达 的 最远距离 。如果一个坐标跟塔的距离在 radius 以内，那么该塔的信号可以到达该坐标。在这个范围以外信号会很微弱，所以 radius 以外的距离该塔是 不能到达的 。

如果第 i 个塔能到达 (x, y) ，那么该塔在此处的信号为 ⌊qi / (1 + d)⌋ ，其中 d 是塔跟此坐标的距离。一个坐标的 信号强度 是所有 能到达 该坐标的塔的信号强度之和。

请你返回数组 [cx, cy] ，表示 信号强度 最大的 整数 坐标点 (cx, cy) 。如果有多个坐标网络信号一样大，请你返回字典序最小的 非负 坐标。

注意：

坐标 (x1, y1) 字典序比另一个坐标 (x2, y2) 小，需满足以下条件之一：
要么 x1 < x2 ，
要么 x1 == x2 且 y1 < y2 。
⌊val⌋ 表示小于等于 val 的最大整数（向下取整函数）。
 

示例 1：


输入：towers = [[1,2,5],[2,1,7],[3,1,9]], radius = 2
输出：[2,1]
解释：
坐标 (2, 1) 信号强度之和为 13
- 塔 (2, 1) 强度参数为 7 ，在该点强度为 ⌊7 / (1 + sqrt(0)⌋ = ⌊7⌋ = 7
- 塔 (1, 2) 强度参数为 5 ，在该点强度为 ⌊5 / (1 + sqrt(2)⌋ = ⌊2.07⌋ = 2
- 塔 (3, 1) 强度参数为 9 ，在该点强度为 ⌊9 / (1 + sqrt(1)⌋ = ⌊4.5⌋ = 4
没有别的坐标有更大的信号强度。
示例 2：

输入：towers = [[23,11,21]], radius = 9
输出：[23,11]
解释：由于仅存在一座信号塔，所以塔的位置信号强度最大。
示例 3：

输入：towers = [[1,2,13],[2,1,7],[0,1,9]], radius = 2
输出：[1,2]
解释：坐标 (1, 2) 的信号强度最大。
```

`思路`
找到towes里最大的x和y,遍历所有点，计算最大的一个

`c# 实现`
```
public class Solution {
    public int[] BestCoordinate(int[][] towers, int radius) {
        int xMax = int.MinValue, yMax = int.MinValue;
        foreach (int[] tower in towers) {
            int x = tower[0], y = tower[1];
            xMax = Math.Max(xMax, x);
            yMax = Math.Max(yMax, y);
        }
        int cx = 0, cy = 0;
        int maxQuality = 0;
        for (int x = 0; x <= xMax; x++) {
            for (int y = 0; y <= yMax; y++) {
                int[] coordinate = {x, y};
                int quality = 0;
                foreach (int[] tower in towers) {
                    int squaredDistance = GetSquaredDistance(coordinate, tower);
                    if (squaredDistance <= radius * radius) {
                        double distance = Math.Sqrt(squaredDistance);
                        quality += (int) Math.Floor(tower[2] / (1 + distance));
                    }
                }
                if (quality > maxQuality) {
                    cx = x;
                    cy = y;
                    maxQuality = quality;
                }
            }
        }
        return new int[]{cx, cy};
    }

    public int GetSquaredDistance(int[] coordinate, int[] tower) {
        return (tower[0] - coordinate[0]) * (tower[0] - coordinate[0]) + (tower[1] - coordinate[1]) * (tower[1] - coordinate[1]);
    }
}
```

***

## 2022/11/1

## 1662. 检查两个字符串数组是否相等

[1662. 检查两个字符串数组是否相等](https://leetcode.cn/problems/check-if-two-string-arrays-are-equivalent/)
```

给你两个字符串数组 word1 和 word2 。如果两个数组表示的字符串相同，返回 true ；否则，返回 false 。

数组表示的字符串 是由数组中的所有元素 按顺序 连接形成的字符串。

 

示例 1：

输入：word1 = ["ab", "c"], word2 = ["a", "bc"]
输出：true
解释：
word1 表示的字符串为 "ab" + "c" -> "abc"
word2 表示的字符串为 "a" + "bc" -> "abc"
两个字符串相同，返回 true
示例 2：

输入：word1 = ["a", "cb"], word2 = ["ab", "c"]
输出：false
示例 3：

输入：word1  = ["abc", "d", "defg"], word2 = ["abcddefg"]
输出：true
```

`思路`
拼接两个字符串，返回是否相同

`c# 实现`
```
public class Solution {
    public bool ArrayStringsAreEqual(string[] word1, string[] word2) {
        string w1 = String.Empty , w2 = String.Empty;
        foreach (string str in word1)
        {
            w1 += str;
        }
        foreach (string str in word2)
        {
            w2 += str;
        }
        return w1 == w2;
    }
}
```

***

## 2022/10/31

## 481. 神奇字符串

[481. 神奇字符串](https://leetcode.cn/problems/magical-string/)
```

神奇字符串 s 仅由 '1' 和 '2' 组成，并需要遵守下面的规则：

神奇字符串 s 的神奇之处在于，串联字符串中 '1' 和 '2' 的连续出现次数可以生成该字符串。
s 的前几个元素是 s = "1221121221221121122……" 。如果将 s 中连续的若干 1 和 2 进行分组，可以得到 "1 22 11 2 1 22 1 22 11 2 11 22 ......" 。每组中 1 或者 2 的出现次数分别是 "1 2 2 1 1 2 1 2 2 1 2 2 ......" 。上面的出现次数正是 s 自身。

给你一个整数 n ，返回在神奇字符串 s 的前 n 个数字中 1 的数目。

 

示例 1：

输入：n = 6
输出：3
解释：神奇字符串 s 的前 6 个元素是 “122112”，它包含三个 1，因此返回 3 。 
示例 2：

输入：n = 1
输出：1
```

`思路`
初始化[1,2,2]的数组，根据题目规则往后生成数字，累积其中的数字1

`c# 实现`
```
public class Solution {
    public int MagicalString(int n) {
         if (n < 4) {
            return 1;
        }
        int[] arr = new int[n];
        arr[0] = 1;
        arr[1] = 2;
        arr[2] = 2;
        int ans = 1 , left = 2 , right = 3;
        while(right < n){
            int cnt = arr[left];
            int num = 3 - arr[right - 1];
            while(cnt > 0 && right < n){
                arr[right] = num;
                if (num == 1)
                {
                    ans ++;
                }
                right ++;
                cnt --;
            }
            left ++;
        }
        return ans;
    }
}
```

***

## 907. 子数组的最小值之和

[907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)
```

给定一个整数数组 arr，找到 min(b) 的总和，其中 b 的范围为 arr 的每个（连续）子数组。

由于答案可能很大，因此 返回答案模 10^9 + 7 。

 

示例 1：

输入：arr = [3,1,2,4]
输出：17
解释：
子数组为 [3]，[1]，[2]，[4]，[3,1]，[1,2]，[2,4]，[3,1,2]，[1,2,4]，[3,1,2,4]。 
最小值为 3，1，2，4，1，1，2，1，1，1，和为 17。
示例 2：

输入：arr = [11,81,94,43,3]
输出：444
```

`思路`
找到数组中每个数作为最小数的次数

`c# 实现`
```
public class Solution {
    public int SumSubarrayMins(int[] arr) {
        int n = arr.Length;
        Stack<int> monoStack = new Stack<int>();
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 0; i < n; i++) {
            while (monoStack.Count > 0 && arr[i] <= arr[monoStack.Peek()]) {
                monoStack.Pop();
            }
            left[i] = i - (monoStack.Count == 0 ? -1 : monoStack.Peek());
            monoStack.Push(i);
        }
        monoStack.Clear();
        for (int i = n - 1; i >= 0; i--) {
            while (monoStack.Count > 0 && arr[i] < arr[monoStack.Peek()]) {
                monoStack.Pop();
            }
            right[i] = (monoStack.Count == 0 ? n : monoStack.Peek()) - i;
            monoStack.Push(i);
        }
        long ans = 0;
        const int MOD = 1000000007;
        for (int i = 0; i < n; i++) {
            ans = (ans + (long) left[i] * right[i] * arr[i]) % MOD; 
        }
        return (int) ans;
    }
}
```

***




## 2022/10/27

## 1822. 数组元素积的符号

[1822. 数组元素积的符号](https://leetcode.cn/problems/sign-of-the-product-of-an-array/)
```

已知函数 signFunc(x) 将会根据 x 的正负返回特定值：

如果 x 是正数，返回 1 。
如果 x 是负数，返回 -1 。
如果 x 是等于 0 ，返回 0 。
给你一个整数数组 nums 。令 product 为数组 nums 中所有元素值的乘积。

返回 signFunc(product) 。

 

示例 1：

输入：nums = [-1,-2,-3,-4,3,2,1]
输出：1
解释：数组中所有值的乘积是 144 ，且 signFunc(144) = 1
示例 2：

输入：nums = [1,5,0,2,-3]
输出：0
解释：数组中所有值的乘积是 0 ，且 signFunc(0) = 0
示例 3：

输入：nums = [-1,1,-1,1,-1]
输出：-1
解释：数组中所有值的乘积是 -1 ，且 signFunc(-1) = -1
```

`思路`
遍历数组，计数所有负数，偶数则返回1

`c# 实现`
```
public class Solution {
    public int ArraySign(int[] nums) {
        int negative = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            if(nums[i] == 0){
                return 0;
            }else(nums[i] < 0)
            {
                negative++;
            }   
        }
        return (negative&1) == 1 ? -1 : 1;
    }
}
```

***

## 2022/10/25

## 934. 最短的桥

[934. 最短的桥](https://leetcode.cn/problems/shortest-bridge/submissions/)
```
给你一个大小为 n x n 的二元矩阵 grid ，其中 1 表示陆地，0 表示水域。
岛 是由四面相连的 1 形成的一个最大组，即不会与非组内的任何其他 1 相连。grid 中 恰好存在两座岛 。
你可以将任意数量的 0 变为 1 ，以使两座岛连接起来，变成 一座岛 。
返回必须翻转的 0 的最小数目。
 
示例 1：
输入：grid = [[0,1],[1,0]]
输出：1
示例 2：
输入：grid = [[0,1,0],[0,0,0],[0,0,1]]
输出：2
示例 3：
输入：grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
输出：1
```

`思路`
找到第一个为1的位置，BFS遍历一次记录第一座岛的所有位置（岛的位置记为-1避免重复）。再BFS第一座岛所有位置，向外找为1的地方，找到返回结果，否则计数并继续。

`c# 实现`
```
public class Solution {
    public int ShortestBridge(int[][] grid) {
        int n = grid.Length;
        int[][] dirs = {new int[]{-1, 0}, new int[]{1, 0}, new int[]{0, 1}, new int[]{0, -1}};
        IList<Tuple<int, int>> island = new List<Tuple<int, int>>();
        Queue<Tuple<int, int>> queue = new Queue<Tuple<int, int>>();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1)
                {
                    queue.Enqueue(new Tuple<int, int>(i, j));
                    grid[i][j] = -1;
                    while(queue.Count > 0){
                        Tuple<int, int> cell = queue.Dequeue();
                        int x = cell.Item1, y = cell.Item2;
                        island.Add(cell);
                        for (int k = 0; k < 4; k++)
                        {
                            int nx = dirs[k][0] + x;
                            int ny = dirs[k][1] + y;
                            if (nx >= 0 && ny >= 0 && nx < n && ny < n && grid[nx][ny] == 1)
                            {
                                queue.Enqueue(new Tuple<int, int>(nx, ny));
                                grid[nx][ny] = -1;
                            }
                        }
                    }

                    foreach (Tuple<int, int> cell in island) {
                        queue.Enqueue(cell);
                    }
                    int ans = 0;
                    while(queue.Count > 0){
                        int sz = queue.Count;
                        for (int k = 0; k < sz; k++)
                        {
                            Tuple<int, int> cell = queue.Dequeue();
                            int x = cell.Item1, y = cell.Item2;
                            for (int l = 0; l < 4; l++)
                            {
                                int nx = dirs[l][0] + x;
                                int ny = dirs[l][1] + y;
                                if (nx >= 0 && ny >= 0 && nx < n && ny < n)
                                {
                                    if (grid[nx][ny] == 1)
                                    {
                                        return ans;
                                    }else if(grid[nx][ny] == 0){
                                        queue.Enqueue(new Tuple<int, int>(nx, ny));
                                        grid[nx][ny] = -1;
                                    }
                                }
                            }
                        }
                        ans ++;
                    }
                }
            }
        }
         return 0;
    }
}
```

***

## 2022/10/24

## 915. 分割数组

[915. 分割数组](https://leetcode.cn/problems/partition-array-into-disjoint-intervals)
```
给定一个数组 nums ，将其划分为两个连续子数组 left 和 right， 使得：

left 中的每个元素都小于或等于 right 中的每个元素。
left 和 right 都是非空的。
left 的长度要尽可能小。
在完成这样的分组后返回 left 的 长度 。

用例可以保证存在这样的划分方法。

 

示例 1：

输入：nums = [5,0,3,8,6]
输出：3
解释：left = [5,0,3]，right = [8,6]
示例 2：

输入：nums = [1,1,1,0,6,12]
输出：4
解释：left = [1,1,1,0]，right = [6,12]
```

`思路`
遍历nums 记录当前最大值curMax和历史最大值leftMax，当nums[i]小于leftMax时更新leftMax与最终位置。

`c# 实现`
```
 public class Solution {
    public int PartitionDisjoint(int[] nums) {
        int n = nums.Length;
        int leftMax = nums[0], leftPos = 0, curMax = nums[0];
        for (int i = 1; i < n; i++)
        {
            curMax = Math.Max(nums[i] ,curMax );
            if (nums[i] < leftMax)
            {
                leftMax = curMax;
                leftPos = i;
            }
        }
        return leftPos + 1;
    }
}
```

***

## 2022/10/21

## 901. 股票价格跨度
股票价格跨度

[901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/)
```
编写一个 StockSpanner 类，它收集某些股票的每日报价，并返回该股票当日价格的跨度。

今天股票价格的跨度被定义为股票价格小于或等于今天价格的最大连续日数（从今天开始往回数，包括今天）。

例如，如果未来7天股票的价格是 [100, 80, 60, 70, 60, 75, 85]，那么股票跨度将是 [1, 1, 1, 2, 1, 4, 6]。

 

示例：

输入：["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
输出：[null,1,1,1,2,1,4,6]
解释：
首先，初始化 S = StockSpanner()，然后：
S.next(100) 被调用并返回 1，
S.next(80) 被调用并返回 1，
S.next(60) 被调用并返回 1，
S.next(70) 被调用并返回 2，
S.next(60) 被调用并返回 1，
S.next(75) 被调用并返回 4，
S.next(85) 被调用并返回 6。

注意 (例如) S.next(75) 返回 4，因为截至今天的最后 4 个价格
(包括今天的价格 75) 小于或等于今天的价格。
```

`思路`
往前寻找比当前小的数的个数。

`c# 实现`
```
public class StockSpanner {
    IList<int> list;
    public StockSpanner() {
        list = new List<int>();
    }
    
    public int Next(int price) {
    
        int ans = 1;
        for (int i = list.Count - 1; i >= 0; i--)
        {
            if (price >= list[i])
            {
                ans++;
            }else{
                break;
            }
        }
        list.Add(price);
        return ans;
    }
}

官方答案：
public class StockSpanner {
    Stack<Tuple<int, int>> stack;
    int idx;

    public StockSpanner() {
        stack = new Stack<Tuple<int, int>>();
        stack.Push(new Tuple<int, int>(-1, int.MaxValue));
        idx = -1;
    }

    public int Next(int price) {
        idx++;
        while (price >= stack.Peek().Item2) {
            stack.Pop();
        }
        int ret = idx - stack.Peek().Item1;
        stack.Push(new Tuple<int, int>(idx, price));
        return ret;
    }
}

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/online-stock-span/solution/gu-piao-jie-ge-kua-du-by-leetcode-soluti-5cm7/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

***

## 2022/10/20

## 779. 第K个语法符号

[779. 第K个语法符号](https://leetcode.cn/problems/k-th-symbol-in-grammar/)
```
我们构建了一个包含 n 行( 索引从 1  开始 )的表。首先在第一行我们写上一个 0。接下来的每一行，将前一行中的0替换为01，1替换为10。

例如，对于 n = 3 ，第 1 行是 0 ，第 2 行是 01 ，第3行是 0110 。
给定行数 n 和序数 k，返回第 n 行中第 k 个字符。（ k 从索引 1 开始）


示例 1:

输入: n = 1, k = 1
输出: 0
解释: 第一行：0
示例 2:

输入: n = 2, k = 1
输出: 0
解释: 
第一行: 0 
第二行: 01
示例 3:

输入: n = 2, k = 2
输出: 1
解释:
第一行: 0
第二行: 01
```

`思路`
递归寻找位置k的上一级是0还是1，且根据当前k的奇偶知道最终结果

`c# 实现`
```
public class Solution {
    public int KthGrammar(int n, int k) {
        if (n == 1)
        {
            return 0;
        }else{
            int pre = KthGrammar(n-1,(int)Math.Ceiling((double)k/2));
            if (pre == 0)
            {
                if ((k&1) == 1 )
                {
                    return 0;
                }else{
                    return 1;
                }
            }else{
                if ((k&1) == 1 )
                {
                    return 1;
                }else{
                    return 0;
                }
            }
        }
    }
}
```

***

## 2022/10/19

## 1700. 无法吃午餐的学生数量

[1700. 无法吃午餐的学生数量](https://leetcode.cn/problems/number-of-students-unable-to-eat-lunch/)
```
学校的自助午餐提供圆形和方形的三明治，分别用数字 0 和 1 表示。所有学生站在一个队列里，每个学生要么喜欢圆形的要么喜欢方形的。 餐厅里三明治的数量与学生的数量相同。所有三明治都放在一个 栈 里，每一轮：
* 如果队列最前面的学生 喜欢 栈顶的三明治，那么会 拿走它 并离开队列。
* 否则，这名学生会 放弃这个三明治 并回到队列的尾部。
这个过程会一直持续到队列里所有学生都不喜欢栈顶的三明治为止。
给你两个整数数组 students 和 sandwiches ，其中 sandwiches[i] 是栈里面第 i  个三明治的类型（i = 0 是栈的顶部）， students[j] 是初始队列里第 j  名学生对三明治的喜好（j = 0 是队列的最开始位置）。请你返回无法吃午餐的学生数量。
 
示例 1：
输入：students = [1,1,0,0], sandwiches = [0,1,0,1]
输出：0 
解释：
- 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [1,0,0,1]。
- 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [0,0,1,1]。
- 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [0,1,1]，三明治栈为 sandwiches = [1,0,1]。
- 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [1,1,0]。
- 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [1,0]，三明治栈为 sandwiches = [0,1]。
- 最前面的学生放弃最顶上的三明治，并回到队列的末尾，学生队列变为 students = [0,1]。
- 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = [1]，三明治栈为 sandwiches = [1]。
- 最前面的学生拿走最顶上的三明治，剩余学生队列为 students = []，三明治栈为 sandwiches = []。
所以所有学生都有三明治吃。
示例 2：
输入：students = [1,1,1,0,0,1], sandwiches = [1,0,0,0,1,1]
输出：3
```

`思路`
模拟学生拿三明治的过程。

`c# 实现`
```
public class Solution {
    public int CountStudents(int[] students, int[] sandwiches) {
        Queue<int> queue = new Queue<int>(students);
        Array.Reverse(sandwiches);
        Stack<int> stack = new Stack<int>(sandwiches);
        int cnt = 0;
        while(cnt < queue.Count && queue.Count > 0){
            if (queue.Peek() == stack.Peek())
            {
                queue.Dequeue();
                stack.Pop();
                cnt = 0;
            }else{
                queue.Enqueue(queue.Dequeue());
                cnt ++;
            }
        }
        return queue.Count;
    }
}

官方解法
public class Solution {
    public int CountStudents(int[] students, int[] sandwiches) {
        int s1 = students.Sum();
        int s0 = students.Length - s1;
        for (int i = 0; i < sandwiches.Length; i++) {
            if (sandwiches[i] == 0 && s0 > 0) {
                s0--;
            } else if (sandwiches[i] == 1 && s1 > 0) {
                s1--;
            } else {
                break;
            }
        }
        return s0 + s1;
    }
}


```

***

## 2022/10/17

## 904. 水果成篮

[904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/submissions/)
```
你正在探访一家农场，农场从左到右种植了一排果树。这些树用一个整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树上的水果 种类 。
你想要尽可能多地收集水果。然而，农场的主人设定了一些严格的规矩，你必须按照要求采摘水果：
* 你只有 两个 篮子，并且每个篮子只能装 单一类型 的水果。每个篮子能够装的水果总量没有限制。
* 你可以选择任意一棵树开始采摘，你必须从 每棵 树（包括开始采摘的树）上 恰好摘一个水果 。采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
* 一旦你走到某棵树前，但水果不符合篮子的水果类型，那么就必须停止采摘。
给你一个整数数组 fruits ，返回你可以收集的水果的 最大 数目。
 
示例 1：
输入：fruits = [1,2,1]
输出：3
解释：可以采摘全部 3 棵树。
示例 2：
输入：fruits = [0,1,2,2]
输出：3
解释：可以采摘 [1,2,2] 这三棵树。
如果从第一棵树开始采摘，则只能采摘 [0,1] 这两棵树。
示例 3：
输入：fruits = [1,2,3,2,2]
输出：4
解释：可以采摘 [2,3,2,2] 这四棵树。
如果从第一棵树开始采摘，则只能采摘 [1,2] 这两棵树。
示例 4：
输入：fruits = [3,3,3,1,2,1,1,2,3,3,4]
输出：5
解释：可以采摘 [1,2,1,1,2] 这五棵树。
```

`思路`
滑动窗口
用left right 表示当前最长区域的边界下标，计算left right 间的的值，当长度超出时，移动left，最后返回最大的值

`c# 实现`
```
public class Solution {
    public int TotalFruit(int[] fruits) {
        Dictionary<int , int> temp = new  Dictionary<int , int>();
        int n = fruits.Length;
        int left = 0 , ans = 0;
        for(int right = 0; right < n ; ++ right){
            temp.TryAdd(fruits[right],0);
            temp[fruits[right]] ++;
            while(temp.Count > 2){
                -- temp[fruits[left]];
                if(temp[fruits[left]] == 0){
                    temp.Remove(fruits[left]);
                }
                ++left;
            }
            ans = Math.Max(right - left + 1 , ans);
        }
        return ans;
    }
}
```

***

## 2022/10/13

##769. 最多能完成排序的块

[769. 最多能完成排序的块](https://leetcode.cn/problems/max-chunks-to-make-sorted/)
```
给定一个长度为 n 的整数数组 arr ，它表示在 [0, n - 1] 范围内的整数的排列。

我们将 arr 分割成若干 块 (即分区)，并对每个块单独排序。将它们连接起来后，使得连接的结果和按升序排序后的原数组相同。

返回数组能分成的最多块数量。

 

示例 1:

输入: arr = [4,3,2,1,0]
输出: 1
解释:
将数组分成2块或者更多块，都无法得到所需的结果。
例如，分成 [4, 3], [2, 1, 0] 的结果是 [3, 4, 0, 1, 2]，这不是有序的数组。
示例 2:

输入: arr = [1,0,2,3,4]
输出: 4
解释:
我们可以把它分成两块，例如 [1, 0], [2, 3, 4]。
然而，分成 [1, 0], [2], [3], [4] 可以得到最多的块数。


```

`思路`
想了好多思路都很复杂，最后看了题解却很巧妙，遍历数组，记录当前最大值，如果当前下标与最大值相同(也就是当前最大值正好在排序后数组的正确位置) 则能分块次数加一。

`c# 实现`
```
public class Solution {
    public int MaxChunksToSorted(int[] arr) {
        int m = 0, res = 0;
        for (int i = 0; i < arr.Length; i++) {
            m = Math.Max(m, arr[i]);
            if (m == i) {
                res++;
            }
        }
        return res;
    }
}
```

***


## 2022/10/12

## 817. 链表组件

[817. 链表组件](https://leetcode.cn/problems/linked-list-components/)

```

给定链表头结点 head，该链表上的每个结点都有一个 唯一的整型值 。同时给定列表 nums，该列表是上述链表中整型值的一个子集。

返回列表 nums 中组件的个数，这里对组件的定义为：链表中一段最长连续结点的值（该值必须在列表 nums 中）构成的集合。

 

示例 1：



输入: head = [0,1,2,3], nums = [0,1,3]
输出: 2
解释: 链表中,0 和 1 是相连接的，且 nums 中不包含 2，所以 [0, 1] 是 nums 的一个组件，同理 [3] 也是一个组件，故返回 2。
示例 2：

 

输入: head = [0,1,2,3,4], nums = [0,3,1,4]
输出: 2
解释: 链表中，0 和 1 是相连接的，3 和 4 是相连接的，所以 [0, 1] 和 [3, 4] 是两个组件，故返回 2。

```

`思路`
遍历链表，当前节点val在nums存在一直到nums找不到为止计数组件数加1，可以用hashset记录一次数组增加寻找速度。

`c# 实现`
```
public class Solution {
    public int NumComponents(ListNode head, int[] nums) {
        Array.Sort(nums);
        bool flag = false;
        int ans = 0;
        while(head != null){
            bool exist = ((IList)nums).Contains(head.val);

            if (exist)
            {
                if (!flag)
                {
                    ans++;
                }
                flag = true;
            }else
            {
                flag = false;
            }
            head = head.next;
        }
        return ans;
    }
}
```

***

## 2022/10/11

## 1790. 仅执行一次字符串交换能否使两个字符串相等

[1790. 仅执行一次字符串交换能否使两个字符串相等](https://leetcode.cn/problems/check-if-one-string-swap-can-make-strings-equal/)

```

给你长度相等的两个字符串 s1 和 s2 。一次 字符串交换 操作的步骤如下：选出某个字符串中的两个下标（不必不同），并交换这两个下标所对应的字符。

如果对 其中一个字符串 执行 最多一次字符串交换 就可以使两个字符串相等，返回 true ；否则，返回 false 。

 

示例 1：

输入：s1 = "bank", s2 = "kanb"
输出：true
解释：例如，交换 s2 中的第一个和最后一个字符可以得到 "bank"
示例 2：

输入：s1 = "attack", s2 = "defend"
输出：false
解释：一次字符串交换无法使两个字符串相等
示例 3：

输入：s1 = "kelb", s2 = "kelb"
输出：true
解释：两个字符串已经相等，所以不需要进行字符串交换
示例 4：

输入：s1 = "abcd", s2 = "dcba"
输出：false
```

`思路`
记录两个字符串不同字符的位置下标，如果不同计数不为0或2，则返回false，否则比较s1[0] s2[1] 且s1[1] s2[0] 是否为相同字符.

`c# 实现`
```
public class Solution {
    public bool AreAlmostEqual(string s1, string s2) {
        List<int> list = new List<int>();
        for (int i = 0; i < s1.Length; i++)
        {
            if (s1[i] != s2[i])
            {
                list.Add(i);
                if (list.Count > 2)
                {
                    return false;
                }
            }
        }
        if (list.Count == 0)
        {
            return true;
        }
        if (list.Count == 1)
        {
            return false;
        }
        return s1[list[0]] == s2[list[1]] && s1[list[1]] == s2[list[0]];
    }
}
```

***

## 2022/10/9

## 856. 括号的分数

[856. 括号的分数](https://leetcode.cn/problems/score-of-parentheses/)

```

给定一个平衡括号字符串 S，按下述规则计算该字符串的分数：

() 得 1 分。
AB 得 A + B 分，其中 A 和 B 是平衡括号字符串。
(A) 得 2 * A 分，其中 A 是平衡括号字符串。
 

示例 1：

输入： "()"
输出： 1
示例 2：

输入： "(())"
输出： 2
示例 3：

输入： "()()"
输出： 2
示例 4：

输入： "(()(()))"
输出： 6
```

`思路`
1.递归拆分计算
遍历s，当匹配到一个'（）'时，如果是字符串末尾，则是（A）的结构，返回中间字符串*2，否则是A+B结构 ，返回递归计算ScoreOfParentheses(A)+ScoreOfParentheses(B)的结果。

2.栈运算
'(' 将0入栈 ')'将栈顶结果+1入栈，最后栈顶即为和。


`c# 实现`
```
1.
public class Solution {
    public int ScoreOfParentheses(string s) {
        if (s.Length == 2)
        {
            return 1;
        }
        int bal = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(')
            {
                bal++;
            }else{
                bal--;
            }
            if (bal == 0)
            {
                if (i == s.Length - 1)
                {
                    return 2*ScoreOfParentheses(s.Substring(1,s.Length - 2));
                }else{
                    return ScoreOfParentheses(s.Substring(0,i+1)) + ScoreOfParentheses(s.Substring(i+1));
                }
            }
        }
    }

}
2.
public class Solution {
    public int ScoreOfParentheses(string s) {
        Stack<int> score = new Stack<int>();
        score.Push(0);
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '(')
            {
                score.Push(0);
            }else{
                int v = score.Pop();
                v = Math.Max(1,2*v);
                int sum = score.Pop() + v;
                score.Push(sum);
            }
        }
        return score.Peek();
    }
}
```

***

## 2022/10/8

## 870. 优势洗牌

[870. 优势洗牌](https://leetcode.cn/problems/advantage-shuffle)

```
给定两个大小相等的数组 nums1 和 nums2，nums1 相对于 nums 的优势可以用满足 nums1[i] > nums2[i] 的索引 i 的数目来描述。

返回 nums1 的任意排列，使其相对于 nums2 的优势最大化。

 

示例 1：

输入：nums1 = [2,7,11,15], nums2 = [1,10,4,11]
输出：[2,11,7,15]
示例 2：

输入：nums1 = [12,24,8,32], nums2 = [13,25,32,11]
输出：[24,32,8,12]

```

`思路`
题目类似于田忌赛马的策略
对nums1 和 nums2的下标数组进行排序，遍历排序后的nums1下标数组，如果当前下标的nums1大于当前nums2,则说明当前nums1的值相较于nums2有优势，ans数组的对应nums2的下标位置赋值，而又因为nums2也是有序的，所有小于等于的情况说明当前nums1这个值对于后面的nums2都没有优势，直接放到最后。

`c# 实现`
```
public class Solution {
    public int[] AdvantageCount(int[] nums1, int[] nums2) {
        int n = nums1.Length;
        int[] idx1 = new int[n];
        int[] idx2 = new int[n];
        for (int i = 0; i < n; i++)
        {
            idx1[i] = i;
            idx2[i] = i;
        }
        Array.Sort(idx1,(i,j)=>{return nums1[i] - nums1[j];});
        Array.Sort(idx2,(i,j)=>{return nums2[i] - nums2[j];});
        int left  = 0;
        int right = n - 1;
        int[] ans = new int[n];
        for (int i = 0; i < n; i++)
        {
            if (nums1[idx1[i]] > nums2[idx2[left]])
            {
                ans[idx2[left]] = nums1[idx1[i]];
                ++ left;
            }else{
                ans[idx2[right]] = nums1[idx1[i]];
                -- right;
            }
        }
        return ans;
    }
}
```

***

## 2022/9/30

## 01.08. 零矩阵

[01.08. 零矩阵](https://leetcode.cn/problems/zero-matrix-lcci/)

``` 
编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。

 

示例 1：

输入：
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出：
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
示例 2：

输入：
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出：
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

`思路`
使用两个一维数组记录矩阵中有0的行和列，遍历矩阵，如果该行/列已经被标记有0，则设置此位置为0

`c#实现`
```
public class Solution {
    public void SetZeroes(int[][] matrix) {
        int[] row = new int[matrix.Length];
        int[] col = new int[matrix[0].Length];
        for(int i = 0 ; i < matrix.Length; i ++){
            for(int j =0 ; j < matrix[i].Length ; j++){
                if(matrix[i][j] == 0){
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }
        for(int i = 0 ; i < matrix.Length; i ++){
            for(int j =0 ; j < matrix[i].Length ; j++){
                if( row[i] == 1 || col[j] == 1){
                    matrix[i][j] = 0;
                }
            }
        }
    }
}
```

***

## 2022/9/28

## 面试题 17.09. 第 k 个数

[面试题 17.09. 第 k 个数](https://leetcode.cn/problems/get-kth-magic-number-lcci/)

```

有些数的素因子只有 3，5，7，请设计一个算法找出第 k 个数。注意，不是必须有这些素因子，而是必须不包含其他的素因子。例如，前几个数按顺序应该是 1，3，5，7，9，15，21。

示例 1:

输入: k = 5

输出: 9
```

`思路`
题目其实是说一个数，他只能由3，5，7的乘法运算得到。先是1*3 1*5 1*7 再是 3*3 3*5 3*7 然后5*3 5*5 5*7 3*3*3 可以用堆结构存所有符合的数。遍历k次，获得第k个数。其中使用[优先级队列 PriorityQueue](https://github.com/h87545645/Blog/blob/main/data-structure/%E4%BC%98%E5%85%88%E7%BA%A7%E9%98%9F%E5%88%97PriorityQueue.md)来模拟堆结构。

`c#实现`
```
public class Solution {
    public int GetKthMagicNumber(int k) {
        int[] factors = {3, 5, 7};
        ISet<long> seen = new HashSet<long>();
        PriorityQueue<long, long> heap = new PriorityQueue<long, long>();
        seen.Add(1);
        heap.Enqueue(1, 1);
        int magic = 0;
        for (int i = 0; i < k; i++) {
            long curr = heap.Dequeue();
            magic = (int) curr;
            foreach (int factor in factors) {
                long next = curr * factor;
                if (seen.Add(next)) {
                    heap.Enqueue(next, next);
                }
            }
        }
        return magic;
    }
}
```

***

## 2022/9/27

## 面试题 01.02. 判定是否互为字符重排

[面试题 01.02. 判定是否互为字符重排](https://leetcode.cn/problems/check-permutation-lcci/)

```
面试题 01.02. 判定是否互为字符重排
给定两个字符串 s1 和 s2，请编写一个程序，确定其中一个字符串的字符重新排列后，能否变成另一个字符串。

示例 1：

输入: s1 = "abc", s2 = "bca"
输出: true 
示例 2：

输入: s1 = "abc", s2 = "bad"
输出: false

```

`思路`
转成字符数组排序后再比较是否相同

`c# 实现`
```
public class Solution {
    public bool CheckPermutation(string s1, string s2) {
        char[] t1 = s1.ToCharArray();
        char[] t2 = s2.ToCharArray();
        Array.Sort(t1);
        Array.Sort(t2);
        return  t1.SequenceEqual(t2);
    }
}
```

***

## 2022/9/23

## 707. 设计链表

[707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

```

设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
```


`c#实现`
```
public class MyLinkedList {
    private Node head;
    private int size;
    public MyLinkedList() {
        size = 0;
        head = new Node(0);
    }
    
    public int Get(int index) {
        Node cur = head;
        int cnt = -1;
        while(cur != null){
            if (cnt != index)
            {
                cnt ++;
                cur = cur.next;
            }else{
                return cur.val;
            }
        }
        return -1;
    }
    
    public void AddAtHead(int val) {
         AddAtIndex(0, val);
    }
    
    public void AddAtTail(int val) {
        AddAtIndex(size, val);
    }

    public void AddAtIndex(int index, int val) {
        if (index > size) {
            return;
        }
        index = Math.Max(0, index);
        size++;
  
        Node pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        Node newNode = new Node(val);
        newNode.next = pred.next;
        pred.next = newNode;
    }
    
    public void DeleteAtIndex(int index) {
         if (index < 0 || index >= size) {
            return;
        }
        size--;
        Node pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        pred.next = pred.next.next;
    }

     class Node {
        public int val;
        public Node next;
        public Node(int val, Node next = null){
            this.val = val;
            this.next = next;
        }
    }
}
```

***

## 2022/9/22

## 1640. 能否连接形成数组

[1640. 能否连接形成数组](https://leetcode.cn/problems/check-array-formation-through-concatenation/)
```

给你一个整数数组 arr ，数组中的每个整数 互不相同 。另有一个由整数数组构成的数组 pieces，其中的整数也 互不相同 。请你以 任意顺序 连接 pieces 中的数组以形成 arr 。但是，不允许 对每个数组 pieces[i] 中的整数重新排序。

如果可以连接 pieces 中的数组形成 arr ，返回 true ；否则，返回 false 。

 

示例 1：

输入：arr = [15,88], pieces = [[88],[15]]
输出：true
解释：依次连接 [15] 和 [88]
示例 2：

输入：arr = [49,18,16], pieces = [[16,18,49]]
输出：false
解释：即便数字相符，也不能重新排列 pieces[0]
示例 3：

输入：arr = [91,4,64,78], pieces = [[78],[4,64],[91]]
输出：true
解释：依次连接 [91]、[4,64] 和 [78]
```

`思路`
先记录下pieces每个数组的第一个元素，再便利arr，如果第一个元素相同则继续比较每个元素

`c# 实现`
```
public class Solution {
    public bool CanFormArray(int[] arr, int[][] pieces) {
        Dictionary<int,int> dict = new Dictionary<int,int>();
        for(int i = 0; i < pieces.Length; i++){
            dict.TryAdd(pieces[i][0],0);
            dict[pieces[i][0]] = i;
        }
        for(int i = 0; i < arr.Length; i++){
            if (dict.ContainsKey(arr[i]))
            {
                int len = pieces[dict[arr[i]]].Length;
                int[] temp = arr.Skip(i).Take(len).ToArray();
                if (Enumerable.SequenceEqual(temp, pieces[dict[arr[i]]]))
                {
                    i += len - 1;
                }else{
                    return false;
                }
            }else{
                return false;
            }
        }
        return true;
    }
}
```

***

## 2022/9/19

## 1636. 按照频率将数组升序排序

[1636. 按照频率将数组升序排序](https://leetcode.cn/problems/sort-array-by-increasing-frequency/)
```
给你一个整数数组 nums ，请你将数组按照每个值的频率 升序 排序。如果有多个值的频率相同，请你按照数值本身将它们 降序 排序。 

请你返回排序后的数组。

 

示例 1：

输入：nums = [1,1,2,2,2,3]
输出：[3,1,1,2,2,2]
解释：'3' 频率为 1，'1' 频率为 2，'2' 频率为 3 。
示例 2：

输入：nums = [2,3,1,3,2]
输出：[1,3,3,2,2]
解释：'2' 和 '3' 频率都为 2 ，所以它们之间按照数值本身降序排序。
示例 3：

输入：nums = [-1,1,-6,4,5,-6,1,4,1]
输出：[5,-1,4,4,-6,-6,1,1,1]

```

`思路`
先记录数组中各个数字出现的次数，再根据出现次数对数组进行排序

`c# 实现`
```
public class Solution {
    public int[] FrequencySort(int[] nums) {
        Dictionary<int, int> cnt = new Dictionary<int, int>();
        foreach (int num in nums) {
            cnt.TryAdd(num, 0);
            cnt[num]++;
        }
        Array.Sort(nums,(int a, int b)=>{
            int cnt1 = cnt[a], cnt2 = cnt[b];
            return cnt1 != cnt2 ? cnt1 - cnt2 : b - a;
        });
        return nums;
    }
}
```

***

## 2022/9/14

## 1619. 删除某些元素后的数组均值

[1619. 删除某些元素后的数组均值](https://leetcode.cn/problems/mean-of-array-after-removing-some-elements/)

```
给你一个整数数组 arr ，请你删除最小 5% 的数字和最大 5% 的数字后，剩余数字的平均值。

与 标准答案 误差在 10-5 的结果都被视为正确结果。

 

示例 1：

输入：arr = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]
输出：2.00000
解释：删除数组中最大和最小的元素后，所有元素都等于 2，所以平均值为 2 。
示例 2：

输入：arr = [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]
输出：4.00000
示例 3：

输入：arr = [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4]
输出：4.77778
示例 4：

输入：arr = [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3]
输出：5.27778
示例 5：

输入：arr = [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]
输出：5.29167
```

`思路`
排序数组，计数中间90%的数字的平均值

`c# 实现`
```
public class Solution {
    public double TrimMean(int[] arr) {
        Array.Sort(arr);
        int len = arr.Length;
        // int rest = (int)Math.Ceiling((double)(len * 0.05));
        double ans = 0;
        // for(int i = rest; i < arr.Length - rest; i++){
        //     ans += arr[i];
        // }
        for(int i = len / 20; i < 19*(len / 20); i++){
            ans += arr[i];
        }
        ans /= (0.9*len);
        return ans;
    }
}
```

***

## 2022/9/9

## 1598. 文件夹操作日志搜集器

[1598. 文件夹操作日志搜集器](https://leetcode.cn/problems/crawler-log-folder/)

```

每当用户执行变更文件夹操作时，LeetCode 文件系统都会保存一条日志记录。

下面给出对变更操作的说明：

"../" ：移动到当前文件夹的父文件夹。如果已经在主文件夹下，则 继续停留在当前文件夹 。
"./" ：继续停留在当前文件夹。
"x/" ：移动到名为 x 的子文件夹中。题目数据 保证总是存在文件夹 x 。
给你一个字符串列表 logs ，其中 logs[i] 是用户在 ith 步执行的操作。

文件系统启动时位于主文件夹，然后执行 logs 中的操作。

执行完所有变更文件夹操作后，请你找出 返回主文件夹所需的最小步数 。

 

示例 1：



输入：logs = ["d1/","d2/","../","d21/","./"]
输出：2
解释：执行 "../" 操作变更文件夹 2 次，即可回到主文件夹
示例 2：



输入：logs = ["d1/","d2/","./","d3/","../","d31/"]
输出：3
示例 3：

输入：logs = ["d1/","../","../","../"]
输出：0
```

`思路`
遍历logs，../计数减1 ./计数不变，其他的计数加1，保证ans不小于0

`c# 实现`
```
public class Solution {
    public int MinOperations(string[] logs) {
        int ans = 0;
        for(int i = 0; i < logs.Length; i++){
            if(logs[i] == "../"){
                ans -= 1;
            }else if(logs[i] == "./"){
                continue;
            }else{
                ans += 1;
            }
            ans = Math.Max(0,ans);
        }
        return ans;
    }
}
```

***


## 2022/9/5

## 652. 寻找重复的子树

[652. 寻找重复的子树](https://leetcode.cn/problems/find-duplicate-subtrees/)

```

给定一棵二叉树 root，返回所有重复的子树。

对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。

如果两棵树具有相同的结构和相同的结点值，则它们是重复的。

 

示例 1：



输入：root = [1,2,3,4,null,2,4,null,null,4]
输出：[[2,4],[4]]
示例 2：



输入：root = [2,1,1]
输出：[[1]]
示例 3：



输入：root = [2,2,2,3,null,3,null]
输出：[[2,3],[3]]
```

`思路`
遍历这个树，用字典记数每个节点的结构字符串，如果某个字符串key的值等于2时，则说明有重复的节点，将其添加到list中。

`c# 实现`
```
public class Solution {
    private Dictionary<string,int> NodeDict;
     IList<TreeNode> ans;
    public IList<TreeNode> FindDuplicateSubtrees(TreeNode root) {
        NodeDict = new Dictionary<string,int>();
        ans = new List<TreeNode>();
        DFS(root);
        return ans;
    }

    string DFS(TreeNode root){
        if (root == null)
        {
            return "null";
        }
        string left = DFS(root.left);
        string right = DFS(root.right);
        string valStr = root.val.ToString() + "," +left +","+ right;
        if (!NodeDict.ContainsKey(valStr))
        {
            NodeDict.Add(valStr,0);
        }
        NodeDict[valStr] ++;
        if (NodeDict[valStr] == 2)
        {
            ans.Add(root);
        }
        return valStr;
    }
}

```

***

## 2022/9/2

## 687. 最长同值路径

[687. 最长同值路径](https://leetcode.cn/problems/longest-univalue-path/)

```

给定一个二叉树的 root ，返回 最长的路径的长度 ，这个路径中的 每个节点具有相同值 。 这条路径可以经过也可以不经过根节点。

两个节点之间的路径长度 由它们之间的边数表示。

 

示例 1:



输入：root = [5,4,5,1,1,5]
输出：2
示例 2:



输入：root = [1,4,5,4,4,5]
输出：2
```

`思路`
递归深度优先遍历树，获得每个节点下的最长路径。

`c# 实现`
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    int res;
    public int LongestUnivaluePath(TreeNode root) {
        res = 0;
        DFS(root);
        return res;
    }

    public int DFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = DFS(root.left), right = DFS(root.right);
        int left1 = 0, right1 = 0;
        if (root.left != null && root.left.val == root.val) {
            left1 = left + 1;
        }
        if (root.right != null && root.right.val == root.val) {
            right1 = right + 1;
        }
        res = Math.Max(res, left1 + right1);
        return Math.Max(left1, right1);
    }
}
```
***

## 2022/8/30

## 946. 验证栈序列

[946. 验证栈序列](https://leetcode.cn/problems/validate-stack-sequences/)

```

给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。

 

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

`思路`
遍历pushed 入栈 同时判断栈顶元素与当前popped相等则出栈，最后栈为空则合法


`c# 实现`
```
public class Solution {
    public bool ValidateStackSequences(int[] pushed, int[] popped) {
        Stack<int> stack = new Stack<int>();
        for (int i = 0 , j = 0; i < pushed.Length; i++)
        {
            stack.Push(pushed[i]);
            while(stack.Count > 0 && stack.Peek() == popped[j]){
                stack.Pop();
                j++;
            }
        }
        return stack.Count == 0;
    }
}
```


***

## 2022/8/30

## 998. 最大二叉树 II

[998. 最大二叉树 II](https://leetcode.cn/problems/maximum-binary-tree-ii/)

```

最大树 定义：一棵树，并满足：其中每个节点的值都大于其子树中的任何其他值。

给你最大树的根节点 root 和一个整数 val 。

就像 之前的问题 那样，给定的树是利用 Construct(a) 例程从列表 a（root = Construct(a)）递归地构建的：

如果 a 为空，返回 null 。
否则，令 a[i] 作为 a 的最大元素。创建一个值为 a[i] 的根节点 root 。
root 的左子树将被构建为 Construct([a[0], a[1], ..., a[i - 1]]) 。
root 的右子树将被构建为 Construct([a[i + 1], a[i + 2], ..., a[a.length - 1]]) 。
返回 root 。
请注意，题目没有直接给出 a ，只是给出一个根节点 root = Construct(a) 。

假设 b 是 a 的副本，并在末尾附加值 val。题目数据保证 b 中的值互不相同。

返回 Construct(b) 。

 

示例 1：



输入：root = [4,1,3,null,null,2], val = 5
输出：[5,4,null,1,3,null,null,2]
解释：a = [1,4,2,3], b = [1,4,2,3,5]
示例 2：


输入：root = [5,2,4,null,1], val = 3
输出：[5,2,4,null,1,null,3]
解释：a = [2,1,5,4], b = [2,1,5,4,3]
示例 3：


输入：root = [5,2,3,null,1], val = 4
输出：[5,2,4,null,1,3]
解释：a = [2,1,5,3], b = [2,1,5,3,4]
 

提示：

树中节点数目在范围 [1, 100] 内
1 <= Node.val <= 100
树中的所有值 互不相同
1 <= val <= 100
```

`思路`
新加的val在数组最右边，所以对于某一节点只有两种情况，如果val大于该节点的值，则这个节点是val节点的字节点，否则应该递归该节点的右子树

`c# 实现`
```
public class Solution {
    TreeNode insertNd = null;
    public TreeNode InsertIntoMaxTree(TreeNode root, int val) {
        if (insertNd == null)
        {
            insertNd = new TreeNode(val);
        }
        if (root == null)
        {
            root = insertNd;
        }else{
            if (val > root.val)
            {
                insertNd.left = root;
                root = insertNd;
            }else{
                root.right = InsertIntoMaxTree(root.right,val);
            }
        }

        return root;
    }
}
```
***

## 2022/8/29

## 1470. 重新排列数组

[1464. 数组中两元素的最大乘积](https://leetcode.cn/problems/shuffle-the-array/)
```
1470. 重新排列数组
给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。

请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。

 

示例 1：

输入：nums = [2,5,1,3,4,7], n = 3
输出：[2,3,5,4,1,7] 
解释：由于 x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 ，所以答案为 [2,3,5,4,1,7]
示例 2：

输入：nums = [1,2,3,4,4,3,2,1], n = 4
输出：[1,4,2,3,3,2,4,1]
示例 3：

输入：nums = [1,1,2,2], n = 2
输出：[1,2,1,2]
```

`思路`
申明一个长度2n的数组，将对应位置的元素放入

`c# 实现`
```
public class Solution {
    public int[] Shuffle(int[] nums, int n) {
        int[] ans = new int[2 * n];
        for (int i = 0; i < n; i++) {
            ans[2 * i] = nums[i];
            ans[2 * i + 1] = nums[i + n];
        }
        return ans;
    }
}
```
***

## 2022/8/26

## 1464. 数组中两元素的最大乘积

[1464. 数组中两元素的最大乘积](https://leetcode.cn/problems/maximum-product-of-two-elements-in-an-array/)

```

给你一个整数数组 nums，请你选择数组的两个不同下标 i 和 j，使 (nums[i]-1)*(nums[j]-1) 取得最大值。

请你计算并返回该式的最大值。

 

示例 1：

输入：nums = [3,4,5,2]
输出：12 
解释：如果选择下标 i=1 和 j=2（下标从 0 开始），则可以获得最大值，(nums[1]-1)*(nums[2]-1) = (4-1)*(5-1) = 3*4 = 12 。 
示例 2：

输入：nums = [1,5,4,5]
输出：16
解释：选择下标 i=1 和 j=3（下标从 0 开始），则可以获得最大值 (5-1)*(5-1) = 16 。
示例 3：

输入：nums = [3,7]
输出：12
```

``

`思路`
排序后返回最大两个
```
public class Solution {
    public int MaxProduct(int[] nums) {
        Array.Sort(nums,(a,b)=>{
            return b - a;
        });
        return (nums[0] - 1)*(nums[1] - 1);
    }
}
```

***

## 2022/8/25

## 658. 找到 K 个最接近的元素

[658. 找到 K 个最接近的元素](https://leetcode.cn/problems/find-k-closest-elements/)

```

给定一个 排序好 的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。

整数 a 比整数 b 更接近 x 需要满足：

|a - x| < |b - x| 或者
|a - x| == |b - x| 且 a < b
 

示例 1：

输入：arr = [1,2,3,4,5], k = 4, x = 3
输出：[1,2,3,4]
示例 2：

输入：arr = [1,2,3,4,5], k = 4, x = -1
输出：[1,2,3,4]
```

`思路`
直接对数组进行排序，返回前k个元素

`C# 实现`
```

public class Solution {
    public IList<int> FindClosestElements(int[] arr, int k, int x) {
        Array.Sort(arr, (a, b) => {
            if (Math.Abs(a - x) != Math.Abs(b - x)) {
                return Math.Abs(a - x) - Math.Abs(b - x);
            } else {
                return a - b;
            }
        });
        int[] closest = arr.Take(k).ToArray();
        Array.Sort(closest);
        return closest.ToList();
    }
}

```

***

## 2022/8/24

## 1460. 通过翻转子数组使两个数组相等

[1460. 通过翻转子数组使两个数组相等](https://leetcode.cn/problems/make-two-arrays-equal-by-reversing-sub-arrays)

```
给你两个长度相同的整数数组 target 和 arr 。每一步中，你可以选择 arr 的任意 非空子数组 并将它翻转。你可以执行此过程任意次。

如果你能让 arr 变得与 target 相同，返回 True；否则，返回 False 。

 

示例 1：

输入：target = [1,2,3,4], arr = [2,4,1,3]
输出：true
解释：你可以按照如下步骤使 arr 变成 target：
1- 翻转子数组 [2,4,1] ，arr 变成 [1,4,2,3]
2- 翻转子数组 [4,2] ，arr 变成 [1,2,4,3]
3- 翻转子数组 [4,3] ，arr 变成 [1,2,3,4]
上述方法并不是唯一的，还存在多种将 arr 变成 target 的方法。
示例 2：

输入：target = [7], arr = [7]
输出：true
解释：arr 不需要做任何翻转已经与 target 相等。
示例 3：

输入：target = [3,7,9], arr = [3,7,11]
输出：false
解释：arr 没有数字 9 ，所以无论如何也无法变成 target 。

```

`思路`
只要两个数组元素一样，就一定能翻转成同一个种顺序，所以对两个数组排序，然后比较是否一样

`c# 实现`
```
public class Solution {
    public bool CanBeEqual(int[] target, int[] arr) {
        Array.Sort(target);
        Array.Sort(arr);
        return target.SequenceEqual(arr);
    }
}
```

***

## 2022/8/19

## 1450. 在既定时间做作业的学生人数

[1450. 在既定时间做作业的学生人数](https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/)
```

给你两个整数数组 startTime（开始时间）和 endTime（结束时间），并指定一个整数 queryTime 作为查询时间。

已知，第 i 名学生在 startTime[i] 时开始写作业并于 endTime[i] 时完成作业。

请返回在查询时间 queryTime 时正在做作业的学生人数。形式上，返回能够使 queryTime 处于区间 [startTime[i], endTime[i]]（含）的学生人数。

 

示例 1：

输入：startTime = [1,2,3], endTime = [3,2,7], queryTime = 4
输出：1
解释：一共有 3 名学生。
第一名学生在时间 1 开始写作业，并于时间 3 完成作业，在时间 4 没有处于做作业的状态。
第二名学生在时间 2 开始写作业，并于时间 2 完成作业，在时间 4 没有处于做作业的状态。
第三名学生在时间 3 开始写作业，预计于时间 7 完成作业，这是是唯一一名在时间 4 时正在做作业的学生。
示例 2：

输入：startTime = [4], endTime = [4], queryTime = 4
输出：1
解释：在查询时间只有一名学生在做作业。
示例 3：

输入：startTime = [4], endTime = [4], queryTime = 5
输出：0
示例 4：

输入：startTime = [1,1,1,1], endTime = [1,3,2,4], queryTime = 7
输出：0
示例 5：

输入：startTime = [9,8,7,6,5,4,3,2,1], endTime = [10,10,10,10,10,10,10,10,10], queryTime = 5
输出：5
```

`思路`
遍历数组 计数小于queryTime的startTime且大于queryTime的endTime

`c# 实现`
```
public class Solution {
    public int BusyStudent(int[] startTime, int[] endTime, int queryTime) {
        int ans = 0;
        for (int i = 0; i < startTime.Length; i++)
        {
            if (queryTime >= startTime[i] && queryTime <= endTime[i])
            {
                ++ans;
            }
        }
        return ans;
    }
}

```

***

## 2022/8/17

## 1302. 层数最深叶子节点的和

[1302. 层数最深叶子节点的和](https://leetcode.cn/problems/deepest-leaves-sum)

```
给你一棵二叉树的根节点 root ，请你返回 层数最深的叶子节点的和 。

 

示例 1：



输入：root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
输出：15
示例 2：

输入：root = [6,7,8,2,7,1,3,9,null,1,4,null,null,null,5]
输出：19
```

`思路`
[DFS](https://github.com/h87545645/Blog/blob/main/algorithm/DFS%26BFS.md)应用题

`c# 实现`
```
public class Solution {
    private int sum, dep;
    public int DeepestLeavesSum(TreeNode root) {
        dep = 0;
        sum = 0;
        DFS(root,0);
        return sum;
    }

    private void DFS(TreeNode root , int level){
        if (root == null)
        {
            return;
        }
        DFS(root.left,level+1);
        if (dep == level)
        {
            sum += root.val;
        }else if(dep < level){
            dep = level;
            sum = 0;
            sum += root.val;
        }
        DFS(root.right,level+1);
    }
}
```

***

## 2022/8/16

## 1656. 设计有序流

[1656. 设计有序流](https://leetcode.cn/problems/design-an-ordered-stream)

```
有 n 个 (id, value) 对，其中 id 是 1 到 n 之间的一个整数，value 是一个字符串。不存在 id 相同的两个 (id, value) 对。

设计一个流，以 任意 顺序获取 n 个 (id, value) 对，并在多次调用时 按 id 递增的顺序 返回一些值。

实现 OrderedStream 类：

OrderedStream(int n) 构造一个能接收 n 个值的流，并将当前指针 ptr 设为 1 。
String[] insert(int id, String value) 向流中存储新的 (id, value) 对。存储后：
如果流存储有 id = ptr 的 (id, value) 对，则找出从 id = ptr 开始的 最长 id 连续递增序列 ，并 按顺序 返回与这些 id 关联的值的列表。然后，将 ptr 更新为最后那个  id + 1 。
否则，返回一个空列表。

 

示例：



输入
["OrderedStream", "insert", "insert", "insert", "insert", "insert"]
[[5], [3, "ccccc"], [1, "aaaaa"], [2, "bbbbb"], [5, "eeeee"], [4, "ddddd"]]
输出
[null, [], ["aaaaa"], ["bbbbb", "ccccc"], [], ["ddddd", "eeeee"]]

解释
OrderedStream os= new OrderedStream(5);
os.insert(3, "ccccc"); // 插入 (3, "ccccc")，返回 []
os.insert(1, "aaaaa"); // 插入 (1, "aaaaa")，返回 ["aaaaa"]
os.insert(2, "bbbbb"); // 插入 (2, "bbbbb")，返回 ["bbbbb", "ccccc"]
os.insert(5, "eeeee"); // 插入 (5, "eeeee")，返回 []
os.insert(4, "ddddd"); // 插入 (4, "ddddd")，返回 ["ddddd", "eeeee"]

```

`思路`
用排序字典SortedDictionary<int,string> dict 记录数据

`c# 实现`
```
public class OrderedStream {
    private int ptr;
    private int capacity;
    private SortedDictionary<int,string> dict;
    public OrderedStream(int n) {
        capacity = n;
        ptr = 1;
        dict = new SortedDictionary<int,string>();
    }
    
    public IList<string> Insert(int idKey, string value) {
        IList<string> ansList= new List<string>();
        dict.Add(idKey,value);
        while(idKey == ptr && dict.ContainsKey(idKey)){
            ansList.Add(dict[idKey]);
            ++ ptr;
            ++ idKey;
        }
        return ansList;
    }
}
```

***

## 2022/8/15

## 641. 设计循环双端队列

[641. 设计循环双端队列](https://leetcode.cn/problems/design-circular-deque)

```
设计实现双端队列。

实现 MyCircularDeque 类:

MyCircularDeque(int k) ：构造函数,双端队列最大为 k 。
boolean insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true ，否则返回 false 。
boolean insertLast() ：将一个元素添加到双端队列尾部。如果操作成功返回 true ，否则返回 false 。
boolean deleteFront() ：从双端队列头部删除一个元素。 如果操作成功返回 true ，否则返回 false 。
boolean deleteLast() ：从双端队列尾部删除一个元素。如果操作成功返回 true ，否则返回 false 。
int getFront() )：从双端队列头部获得一个元素。如果双端队列为空，返回 -1 。
int getRear() ：获得双端队列的最后一个元素。 如果双端队列为空，返回 -1 。
boolean isEmpty() ：若双端队列为空，则返回 true ，否则返回 false  。
boolean isFull() ：若双端队列满了，则返回 true ，否则返回 false 。
 

示例 1：

输入
["MyCircularDeque", "insertLast", "insertLast", "insertFront", "insertFront", "getRear", "isFull", "deleteLast", "insertFront", "getFront"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
输出
[null, true, true, true, false, 2, true, true, true, 4]

解释
MyCircularDeque circularDeque = new MycircularDeque(3); // 设置容量大小为3
circularDeque.insertLast(1);			        // 返回 true
circularDeque.insertLast(2);			        // 返回 true
circularDeque.insertFront(3);			        // 返回 true
circularDeque.insertFront(4);			        // 已经满了，返回 false
circularDeque.getRear();  				// 返回 2
circularDeque.isFull();				        // 返回 true
circularDeque.deleteLast();			        // 返回 true
circularDeque.insertFront(4);			        // 返回 true
circularDeque.getFront();				// 返回 4
 

```

`思路`
使用双向链表实现

`c# 实现`
```
public class MyCircularDeque {
    private Node front;
    private Node rear;
    private int capacity;
    private int size;
    public MyCircularDeque(int k) {
        capacity = k;
        size = 0;
    }
    
    public bool InsertFront(int value) {
        if (IsFull())
        {
            return false;
        }
        Node temp = new Node(value);
        if (IsEmpty())
        {
            front = rear = temp;
        }else{
            temp.next = front;
            front.prev = temp;
            front = temp;
        }
        ++ size;
        return true;
    }
    
    public bool InsertLast(int value) {
        if (IsFull())
        {
            return false;
        }
        Node temp = new Node(value);
        if (IsEmpty())
        {
            front = rear = temp;
        }else{
            rear.next = temp;
            temp.prev = rear;
            rear = temp;
        }
        ++ size;
        return true;
    }
    
    public bool DeleteFront() {
        if (IsEmpty())
        {
            return false;
        }
        front = front.next;
        if (front != null) {
            front.prev = null;
        }
        -- size;
        return true;
    }
    
    public bool DeleteLast() {
        if (IsEmpty())
        {
            return false;
        }
        rear = rear.prev;
        if (rear != null) {
            rear.next = null;
        }
        -- size;
        return true;
    }
    
    public int GetFront() {
        if (IsEmpty())
        {
            return -1;
        }
        return front.val;
    }
    
    public int GetRear() {
        if (IsEmpty())
        {
            return -1;
        }
        return rear.val;
    }
    
    public bool IsEmpty() {
        return size == 0;
    }
    
    public bool IsFull() {
        return size == capacity;
    }


     class Node {
        public int val;
        public Node next;
        public Node prev;
        public Node(int val){
            this.val = val;
        }
    }
}
```

***

## 2022/8/12

## 1282. 用户分组

[1282. 用户分组](https://leetcode.cn/problems/group-the-people-given-the-group-size-they-belong-to)

```
有 n 个人被分成数量未知的组。每个人都被标记为一个从 0 到 n - 1 的唯一ID 。

给定一个整数数组 groupSizes ，其中 groupSizes[i] 是第 i 个人所在的组的大小。例如，如果 groupSizes[1] = 3 ，则第 1 个人必须位于大小为 3 的组中。

返回一个组列表，使每个人 i 都在一个大小为 groupSizes[i] 的组中。

每个人应该 恰好只 出现在 一个组 中，并且每个人必须在一个组中。如果有多个答案，返回其中 任何 一个。可以 保证 给定输入 至少有一个 有效的解。

 

示例 1：

输入：groupSizes = [3,3,3,3,3,1,3]
输出：[[5],[0,1,2],[3,4,6]]
解释：
第一组是 [5]，大小为 1，groupSizes[5] = 1。
第二组是 [0,1,2]，大小为 3，groupSizes[0] = groupSizes[1] = groupSizes[2] = 3。
第三组是 [3,4,6]，大小为 3，groupSizes[3] = groupSizes[4] = groupSizes[6] = 3。 
其他可能的解决方案有 [[2,1,6],[5],[0,4,3]] 和 [[5],[0,6,2],[4,3,1]]。
示例 2：

输入：groupSizes = [2,1,3,3,3,2]
输出：[[1],[0,5],[2,3,4]]
```

`思路`
遍历groupSizes 用SortedDictionaryy<int , List<int>>字典记录当前值和下标，key为遍历groupSizes的值，当对应key的list长度大于等于groupSizes的值时，存入ansList并伤处这个key

`c# 实现`
```
public class Solution {
    public IList<IList<int>> GroupThePeople(int[] groupSizes) {
        IList<IList<int>> ansList = new List<IList<int>>();
        SortedDictionary<int , List<int>> indexDict = new SortedDictionary<int , List<int>>();
        for (int i = 0; i < groupSizes.Length; i++)
        {
            if (!indexDict.ContainsKey(groupSizes[i]))
            {
                List<int> list = new List<int>();
                list.Add(i);
                indexDict.Add(groupSizes[i],list);
            }else{
                indexDict[groupSizes[i]].Add(i);
            }
            if (indexDict[groupSizes[i]].Count >= groupSizes[i])
            {
                List<int> temp;
                indexDict.Remove(groupSizes[i],out temp);
                ansList.Add(temp);
                // ansList.Add(new List<int>(indexDict[groupSizes[i]]));
                // indexDict[groupSizes[i]].Clear();
            }
        }
        return ansList;
    }
}
```

***

## 2022/8/11

## 1417. 重新格式化字符串

[1417. 重新格式化字符串](https://leetcode.cn/problems/reformat-the-string)

```
给你一个混合了数字和字母的字符串 s，其中的字母均为小写英文字母。

请你将该字符串重新格式化，使得任意两个相邻字符的类型都不同。也就是说，字母后面应该跟着数字，而数字后面应该跟着字母。

请你返回 重新格式化后 的字符串；如果无法按要求重新格式化，则返回一个 空字符串 。

 

示例 1：

输入：s = "a0b1c2"
输出："0a1b2c"
解释："0a1b2c" 中任意两个相邻字符的类型都不同。 "a0b1c2", "0a1b2c", "0c2a1b" 也是满足题目要求的答案。
示例 2：

输入：s = "leetcode"
输出：""
解释："leetcode" 中只有字母，所以无法满足重新格式化的条件。
示例 3：

输入：s = "1229857369"
输出：""
解释："1229857369" 中只有数字，所以无法满足重新格式化的条件。
示例 4：

输入：s = "covid2019"
输出："c2o0v1i9d"
示例 5：

输入：s = "ab123"
输出："1a2b3"
```

`思路 贪心`
定义字符数组 `Char[] ans` , 字符计数`charCnt`   数字计数`numCnt`。 ans长度为s.Length*2+1方便操作
遍历字符串，将数字放在ans的下标1，3，5... 位置上 非数字放在2，4，6...位置上
如果charCnt 与 numCnt相差大于1，则不满足条件
因为我们第一个放的是数字 所以如果最后charCnt>numCnt 需要将最后位置上的非数字放到第0个位置
最后转成字符串并去掉空字符


`c# 实现`
```
public class Solution {
    public string Reformat(string s) {
        Char[] ans = new Char[s.Length*2+1];
        int charCnt = 0,numCnt = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (Char.IsDigit(s[i]))
            {
                ans[numCnt*2+1] = s[i];
                ++ numCnt;
            }else{

                ans[charCnt*2+2] = s[i];
                ++ charCnt;
            }
        }
        if (Math.Abs(charCnt - numCnt) > 1)
        {
            return "";
        }
        if (charCnt >  numCnt)
        {
             int idx = (charCnt-1)*2+2;
            ans[0] = ans[idx];
            ans[idx] = '\0';
        }
        return string.Join("", ans).Replace("\u0000" , string.Empty);
    }
}
```

***

## 2022/8/10

## 640. 求解方程

[640. 求解方程](https://leetcode.cn/problems/solve-the-equation/)

```
640. 求解方程
求解一个给定的方程，将x以字符串 "x=#value" 的形式返回。该方程仅包含 '+' ， '-' 操作，变量 x 和其对应系数。

如果方程没有解，请返回 "No solution" 。如果方程有无限解，则返回 “Infinite solutions” 。

题目保证，如果方程中只有一个解，返回值 'x' 是一个整数。

 

示例 1：

输入: equation = "x+5-3+x=6+x-2"
输出: "x=2"
示例 2:

输入: equation = "x=x"
输出: "Infinite solutions"
示例 3:

输入: equation = "2x=x"
输出: "x=0"
```

`思路`
模拟正常解方程，计数所有x `xCnt`，计算所有数字和`res`,方程=号左边的数字取负，=号右边的x取负，最后如果`xCnt`和`res`都为0，则无限解。`xCnt`为0`res`不为0则无解，返回`res/xCnt`

`c# 实现`
```
public class Solution {
    public string SolveEquation(string equation) {

        int res = 0;
        int xCnt = 0;
        int flag = -1;

        for (int i = 0; i < equation.Length; i++)
        {
          
            if (Char.IsDigit(equation[i]) || equation[i] == 'x')
            {   
                int tempFlag = 1;
                if (i > 0)
                {
                    if (equation[i - 1] == '-')
                    {
                        tempFlag = -1;
                    }
                }
                int t = equation[i] == 'x' ? 1 : 0;
                while(i < equation.Length && char.IsDigit(equation[i])){
                    t = t * 10 + equation[i] - '0';
                    i++;
                }
                if (i < equation.Length)
                {
                    if (equation[i] == 'x')
                    {
                        xCnt += t * tempFlag * -flag;
                    }else{
                        res += t * tempFlag * flag;
                    }
                }else{
                    res += t * tempFlag * flag;
                }
            }
            if (i < equation.Length && equation[i] == '=')
            {
                flag = 1;
            }
        }
        if (xCnt == 0 && res == 0)
        {
            return "Infinite solutions";
        }
        if (xCnt == 0 && res != 0)
        {
            return "No solution";
        }
        res /= xCnt;
        return "x="+res;
    }
}
```

***


## 2022/8/9

## 1413. 逐步求和得到正数的最小值

[1413. 逐步求和得到正数的最小值](https://leetcode.cn/problems/minimum-value-to-get-positive-step-by-step-sum)

```
给你一个整数数组 nums 。你可以选定任意的 正数 startValue 作为初始值。

你需要从左到右遍历 nums 数组，并将 startValue 依次累加上 nums 数组中的值。

请你在确保累加和始终大于等于 1 的前提下，选出一个最小的 正数 作为 startValue 。

 

示例 1：

输入：nums = [-3,2,-3,4,2]
输出：5
解释：如果你选择 startValue = 4，在第三次累加时，和小于 1 。
                累加求和
                startValue = 4 | startValue = 5 | nums
                  (4 -3 ) = 1  | (5 -3 ) = 2    |  -3
                  (1 +2 ) = 3  | (2 +2 ) = 4    |   2
                  (3 -3 ) = 0  | (4 -3 ) = 1    |  -3
                  (0 +4 ) = 4  | (1 +4 ) = 5    |   4
                  (4 +2 ) = 6  | (5 +2 ) = 7    |   2
示例 2：

输入：nums = [1,2]
输出：1
解释：最小的 startValue 需要是正数。
示例 3：

输入：nums = [1,-2,-3]
输出：5

```

`思路`
只需保证累加的`sum`+`ans`>=1 即可，先计算出累加过程中最小的值，即可知道`ans`


`c# 实现`
```
public class Solution {
    public int MinStartValue(int[] nums) {
        int ans = 1 , sum = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            sum += nums[i];
            ans = Math.Min(ans,sum);
        }
        if (ans < 1)
        {
            ans = 1 - ans;
        }
        return ans;
    }
}
```

***

## 2022/8/8

## 761. 特殊的二进制序列

[761. 特殊的二进制序列](https://leetcode.cn/problems/special-binary-string)

```
特殊的二进制序列是具有以下两个性质的二进制序列：

0 的数量与 1 的数量相等。
二进制序列的每一个前缀码中 1 的数量要大于等于 0 的数量。
给定一个特殊的二进制序列 S，以字符串形式表示。定义一个操作 为首先选择 S 的两个连续且非空的特殊的子串，然后将它们交换。（两个子串为连续的当且仅当第一个子串的最后一个字符恰好为第二个子串的第一个字符的前一个字符。)

在任意次数的操作之后，交换后的字符串按照字典序排列的最大的结果是什么？

示例 1:

输入: S = "11011000"
输出: "11100100"
解释:
将子串 "10" （在S[1]出现） 和 "1100" （在S[3]出现）进行交换。
这是在进行若干次操作后按字典序排列最大的结果。
```

`思路`
这道题目理解起来比较难，看了评论说其实就是找匹配的1和0 把1多的往前放，遍历字符串，当1和0数量相同时则找到了一个匹配的子串，将其放入list中，其字串也要执行相同的操作，最后排序所有匹配的1和0。

`c# 实现`
```
public class Solution {
    public string MakeLargestSpecial(string s) {
        if (s.Length <= 2)
        {
            return s;
        }
        List<String> list = new List<String>();
        int count = 0, prev = 0;
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '1')
            {
                ++count;
            }else{
                --count;
                if (count == 0)
                {
                    list.Add("1"+MakeLargestSpecial(s.Substring(prev+1 , i - prev - 1 ))+"0");
                    prev = i + 1;
                }
            }
        }
        list.Sort((a, b) => b.CompareTo(a));
        StringBuilder ans = new StringBuilder();
        foreach (string sub in list) {
            ans.Append(sub);
        }
        return ans.ToString();
    }

}
```

***

## 2022/8/5

## 623. 在二叉树中增加一行

[623. 在二叉树中增加一行. 非递增顺序的最小子序列](https://leetcode.cn/problems/add-one-row-to-tree)

```
给定一个二叉树的根 root 和两个整数 val 和 depth ，在给定的深度 depth 处添加一个值为 val 的节点行。

注意，根节点 root 位于深度 1 。

加法规则如下:

给定整数 depth，对于深度为 depth - 1 的每个非空树节点 cur ，创建两个值为 val 的树节点作为 cur 的左子树根和右子树根。
cur 原来的左子树应该是新的左子树根的左子树。
cur 原来的右子树应该是新的右子树根的右子树。
如果 depth == 1 意味着 depth - 1 根本没有深度，那么创建一个树节点，值 val 作为整个原始树的新根，而原始树就是新根的左子树。
 

示例 1:



输入: root = [4,2,6,3,1,5], val = 1, depth = 2
输出: [4,1,1,2,null,null,6,3,1,5]
示例 2:



输入: root = [4,2,null,3,1], val = 1, depth = 3
输出:  [4,2,null,1,1,3,null,null,1]

```

`思路`
二叉树的遍历应用，[BFS遍历](https://github.com/h87545645/Blog/blob/main/algorithm/DFS%26BFS.md)二叉树，记录当前深度`curDeep`,当`curDeep == depth - 1`时，创建两个新TreeNode，其左右分别是原节点的左右节点，并将原节点的左右指向这两个新节点。

`c# 实现`
```
public class Solution {
    public TreeNode AddOneRow(TreeNode root, int val, int depth) {
        if (depth == 1)
        {
            TreeNode newRoot = new TreeNode(val,root);
            return newRoot;
        }else{
            Queue nodeQue = new Queue();
            nodeQue.Enqueue(root);
            nodeQue.Enqueue(null);
            int curDeep = 1;
            while(nodeQue.Count > 0){
                TreeNode curNode = nodeQue.Dequeue() as TreeNode;
                if (curNode == null)
                {
                    if (nodeQue.Count == 0)
                    {
                        break;
                    }
                    nodeQue.Enqueue(null);
                    curDeep ++;
                    continue;
                }
                if (curDeep == depth - 1)
                {
                    TreeNode insetLeft = new TreeNode(val,curNode.left);
                    TreeNode insetRight = new TreeNode(val,null,curNode.right);
                    curNode.left = insetLeft;
                    curNode.right = insetRight;
                }else if(curDeep > depth - 1){
                    break;
                }
                if (curNode.left != null)
                {
                     nodeQue.Enqueue(curNode.left);
                }
                if (curNode.right != null)
                {
                     nodeQue.Enqueue(curNode.right);
                }
            }
            return root;
        }
    }
}
```

***

## 2022/8/4

## 1403. 非递增顺序的最小子序列

[原题地址  1403. 非递增顺序的最小子序列](https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/)

```
给你一个数组 nums，请你从中抽取一个子序列，满足该子序列的元素之和 严格 大于未包含在该子序列中的各元素之和。

如果存在多个解决方案，只需返回 长度最小 的子序列。如果仍然有多个解决方案，则返回 元素之和最大 的子序列。

与子数组不同的地方在于，「数组的子序列」不强调元素在原数组中的连续性，也就是说，它可以通过从数组中分离一些（也可能不分离）元素得到。

注意，题目数据保证满足所有约束条件的解决方案是 唯一 的。同时，返回的答案应当按 非递增顺序 排列。

 

示例 1：

输入：nums = [4,3,10,9,8]
输出：[10,9] 
解释：子序列 [10,9] 和 [10,8] 是最小的、满足元素之和大于其他各元素之和的子序列。但是 [10,9] 的元素之和最大。 
示例 2：

输入：nums = [4,4,7,6,7]
输出：[7,7,6] 
解释：子序列 [7,7] 的和为 14 ，不严格大于剩下的其他元素之和（14 = 4 + 4 + 6）。因此，[7,6,7] 是满足题意的最小子序列。注意，元素按非递增顺序返回。  
示例 3：

输入：nums = [6]
输出：[6]

```

`思路`
对`nums`排序并求和`sum`，倒叙遍历`nums`，累加每次遍历的和`ansSum`，累减`sum`，并添加到返回的list中，当`ansSum`大于`sum`时，返回结果，

`c# 实现`
```
public class Solution {
    public IList<int> MinSubsequence(int[] nums) {
        IList<int> ans = new List<int>();
        Array.Sort(nums);
        int sum = 0,ansSum = 0;
        sum = nums.Sum();
        // for (int i = 0; i < nums.Length; i++)
        // {
        //     sum += nums[i];
        // }
        for (int i = nums.Length - 1; i >= 0; i--)
        {
            ans.Add(nums[i]);
            ansSum += nums[i];
            sum -= nums[i];
            if (ansSum > sum)
            {
                break;
            }
        }
        return ans;
    }
}
```

***

## 2022/8/3

## 899. 有序队列

[原题地址  899. 有序队列](https://leetcode.cn/problems/orderly-queue)

```
给定一个字符串 s 和一个整数 k 。你可以从 s 的前 k 个字母中选择一个，并把它加到字符串的末尾。

返回 在应用上述步骤的任意数量的移动后，字典上最小的字符串 。

 

示例 1：

输入：s = "cba", k = 1
输出："acb"
解释：
在第一步中，我们将第一个字符（“c”）移动到最后，获得字符串 “bac”。
在第二步中，我们将第一个字符（“b”）移动到最后，获得最终结果 “acb”。
示例 2：

输入：s = "baaca", k = 3
输出："aaabc"
解释：
在第一步中，我们将第一个字符（“b”）移动到最后，获得字符串 “aacab”。
在第二步中，我们将第三个字符（“c”）移动到最后，获得最终结果 “aaabc”。

```

`思路`
当k大于1时，一定可以将字符串换成从小到大的顺序，而k为1时，遍历string,记录每一次的交换结果，返回最小的那个

`c# 实现`
```
public class Solution {
    public string OrderlyQueue(string s, int k) {
        if (k == 1)
        {
            string shortS = s;
            StringBuilder sbuild = new StringBuilder(s);
            int n = s.Length;
            for (int i = 1; i < n; i++)
            {
                char head = sbuild[0];
                sbuild.Remove(0,1);
                sbuild.Append(head);
                if (sbuild.ToString().CompareTo(shortS) < 0)
                {
                    shortS =  sbuild.ToString();
                }
            }
            return shortS;
        }else{
            char[] sa = s.ToCharArray();
            Array.Sort(sa);
            return new string(sa);
        }
        
    }
}
```

***

## 2022/8/2

## 622. 设计循环队列

[原题地址  622. 设计循环队列](https://leetcode.cn/problems/design-circular-queue)

```
设计你的循环队列实现。 循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

你的实现应该支持如下操作：

MyCircularQueue(k): 构造器，设置队列长度为 k 。
Front: 从队首获取元素。如果队列为空，返回 -1 。
Rear: 获取队尾元素。如果队列为空，返回 -1 。
enQueue(value): 向循环队列插入一个元素。如果成功插入则返回真。
deQueue(): 从循环队列中删除一个元素。如果成功删除则返回真。
isEmpty(): 检查循环队列是否为空。
isFull(): 检查循环队列是否已满。
 

示例：

MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
circularQueue.enQueue(1);  // 返回 true
circularQueue.enQueue(2);  // 返回 true
circularQueue.enQueue(3);  // 返回 true
circularQueue.enQueue(4);  // 返回 false，队列已满
circularQueue.Rear();  // 返回 3
circularQueue.isFull();  // 返回 true
circularQueue.deQueue();  // 返回 true
circularQueue.enQueue(4);  // 返回 true
circularQueue.Rear();  // 返回 4

```

`思路`

链表实现

`c# 实现`
```
public class MyCircularQueue {
    private Node head;
    private Node tail;
    private int capacity;
    private int size;
    public MyCircularQueue(int k) {
        capacity = k;
        size = 0;
    }
    
    public bool EnQueue(int value) {
        if (IsFull()) {
            return false;
        }
        Node temp = new Node(value);
        if (size == 0)
        {
            head = temp;
            tail = temp;
        }else{
            // temp.next = head;
            tail.next = temp;
            tail = temp;
        }
        size ++;
        return true;
    }
    
    public bool DeQueue() {
        if(IsEmpty()){
            return false;
        }
        // tail.next = head.next;
        head = head.next;
        size --;
        return true;
    }
    
    public int Front() {
        if (IsEmpty()) {
            return -1;
        }
        return head.val;
    }
    
    public int Rear() {
        if (IsEmpty()) {
            return -1;
        }
        return tail.val;
    }
    
    public bool IsEmpty() {
        return size == 0;
    }
    
    public bool IsFull() {
        return size == capacity;
    }

    class Node {
        public int val;
        public Node next;
        public Node(int val, Node next = null){
            this.val = val;
            this.next = next;
        }
    }
}
```

***

## 2022/8/1

## 1374. 生成每种字符都是奇数个的字符串

[原题地址 1374. 生成每种字符都是奇数个的字符串](https://leetcode.cn/problems/generate-a-string-with-characters-that-have-odd-counts)
```
给你一个整数 n，请你返回一个含 n 个字符的字符串，其中每种字符在该字符串中都恰好出现 奇数次 。

返回的字符串必须只含小写英文字母。如果存在多个满足题目要求的字符串，则返回其中任意一个即可。

 

示例 1：

输入：n = 4
输出："pppz"
解释："pppz" 是一个满足题目要求的字符串，因为 'p' 出现 3 次，且 'z' 出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ohhh" 和 "love"。
示例 2：

输入：n = 2
输出："xy"
解释："xy" 是一个满足题目要求的字符串，因为 'x' 和 'y' 各出现 1 次。当然，还有很多其他字符串也满足题目要求，比如："ag" 和 "ur"。
示例 3：

输入：n = 7
输出："holasss"

```

`思路`
判断n的奇偶，返回n个字符或者1个字符+n-1个另外字符

`c# 实现`
```
public class Solution {
    public string GenerateTheString(int n) {
        StringBuilder ansBuilder = new StringBuilder();
        int off = 0;

        if ((n&1) == 0)
        {
            off = 1;
            ansBuilder.Append('z');
        }
        for (int i = 0; i < n - off; i++)
        {
            ansBuilder.Append('x');
        }
        return ansBuilder.ToString();
    }
}
```

***

## 2022/7/29

## 593. 有效的正方形

[原题地址 593. 有效的正方形](https://leetcode.cn/problems/valid-square)
```
给定2D空间中四个点的坐标 p1, p2, p3 和 p4，如果这四个点构成一个正方形，则返回 true 。

点的坐标 pi 表示为 [xi, yi] 。输入 不是 按任何顺序给出的。

一个 有效的正方形 有四条等边和四个等角(90度角)。

 

示例 1:

输入: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
输出: True
示例 2:

输入：p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,12]
输出：false
示例 3:

输入：p1 = [1,0], p2 = [-1,0], p3 = [0,1], p4 = [0,-1]
输出：true
```

`思路`
先对四个点进行排序，方便求边长，以左下和右上两点计算，两个边长必须大于0且相等，再计算两边的夹角，我用的勾股定理来判断是否为直角三角形，都满足则返回true

`c# 实现`
```
public class Solution {
    public bool ValidSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int [][] pos = new int[][]{p1,p2,p3,p4};
        //对四个点排序，从左到右 从下到上
        Array.Sort(pos,new CompareMethod());
        //求两边长是否一样
        double dis1 = GetDistance(pos[0],pos[1]);
        double dis2 = GetDistance(pos[0],pos[2]);
        if(dis1 <= 0 || dis2 <= 0 || dis1 != dis2){
            return false;
        }
        double dis3 = GetDistance(pos[3],pos[2]);
        double dis4 = GetDistance(pos[3],pos[1]);
               if(dis3 <= 0 || dis4 <= 0 || dis3 != dis4){
            return false;
        }
        double dis5 = GetDistance(pos[1],pos[2]);
        bool angel = Math.Round(dis1*dis1 + dis2*dis2,2) ==  Math.Round(dis5*dis5,2);
        if (!angel)
        {
            return false;
        }
                // return true;
         angel = Math.Round(dis3*dis3 + dis4*dis4,2) ==  Math.Round(dis5*dis5,2);

        if (!angel)
        {
            return false;
        }
        return true;
    }

    private double GetDistance(int[] a, int[] b)
    {
        int x=Math.Abs(b[0]   -   a[0] );
        int y=Math.Abs(b[1]   -   a[1] );
        return Math.Sqrt(x*x+y*y);
    }

    public class CompareMethod : IComparer<int[]>  //继承IComparer<T>接口，T为要比较的元素的类型
    {                                             //类中类，也可以放在类外面
        public  int Compare(int[] x, int[] y)
        {
            if (x[0] == y[0])
            {
                return x[1] - y[1];
            }else{ 
                return x[0] - y[0];
            }
        }
    }
}
```

***

## 2022/7/28

## 1331. 数组序号转换

[原题地址 1331. 数组序号转换](https://leetcode.cn/problems/rank-transform-of-an-array)

```
给你一个整数数组 arr ，请你将数组中的每个元素替换为它们排序后的序号。

序号代表了一个元素有多大。序号编号的规则如下：

序号从 1 开始编号。
一个元素越大，那么序号越大。如果两个元素相等，那么它们的序号相同。
每个数字的序号都应该尽可能地小。
 

示例 1：

输入：arr = [40,10,20,30]
输出：[4,1,2,3]
解释：40 是最大的元素。 10 是最小的元素。 20 是第二小的数字。 30 是第三小的数字。
示例 2：

输入：arr = [100,100,100]
输出：[1,1,1]
解释：所有元素有相同的序号。
示例 3：

输入：arr = [37,12,28,9,100,56,80,5,12]
输出：[5,3,4,2,8,6,7,1,3]

```

`思路`
拷贝arr并排序，用一个不重复的集合记录每个元素对应的排序，遍历`arr`,将元素对应的排序赋值给`ans`数组

`c# 实现`

```
public class Solution {
    public int[] ArrayRankTransform(int[] arr) {
        int[] sortedArr = new int[arr.Length];
        Array.Copy(arr, 0, sortedArr, 0, arr.Length);
        Array.Sort(sortedArr);
        Dictionary<int, int> ranks = new Dictionary<int, int>();
        int[] ans = new int[arr.Length];
        foreach (int a in sortedArr) {
            ranks.TryAdd(a, ranks.Count + 1);
        }
        for (int i = 0; i < arr.Length; i++) {
            ans[i] = ranks[arr[i]];
        }
        return ans;
    }
}
```

***

## 2022/7/27

## 592. 分数加减运算

[原题地址 592. 分数加减运算](https://leetcode.cn/problems/fraction-addition-and-subtraction)

```
给定一个表示分数加减运算的字符串 expression ，你需要返回一个字符串形式的计算结果。 

这个结果应该是不可约分的分数，即最简分数。 如果最终结果是一个整数，例如 2，你需要将它转换成分数形式，其分母为 1。所以在上述例子中, 2 应该被转换为 2/1。

 

示例 1:

输入: expression = "-1/2+1/2"
输出: "0/1"
 示例 2:

输入: expression = "-1/2+1/2+1/3"
输出: "1/3"
示例 3:

输入: expression = "1/3-1/2"
输出: "-1/6"

```

`思路`
模拟运算，遍历`expression`，遇到运算符则将左右两边分数计算，并约分，记录结果，直到计算完成。

`c# 实现`
```
public class Solution {
    public string FractionAddition(string expression) {
        string ans = "";
        for(int i = 0; i < expression.Length; i++){
            if(i > 0){
                if(expression[i] == '+' || expression[i] == '-'){
                    
                    int nextOperator1 = expression.IndexOf('+',i+1,expression.Length - 1 -(i+1));
                    int nextOperator2 = expression.IndexOf('-',i+1,expression.Length - 1- (i+1));
                    int nextOperator;
                    if(nextOperator1 == -1 && nextOperator2 == -1){
                        nextOperator = expression.Length;
                    }else{
                        if(nextOperator1 == -1){
                            nextOperator = nextOperator2;
                        }
                        else if(nextOperator2 == -1){
                            nextOperator = nextOperator1;
                        }else{
                            nextOperator = Math.Min(nextOperator1,nextOperator2);
                        }
                    }
                    string fraction2 = expression.Substring(i + 1,nextOperator - i -1 );
                    string fraction1;
                    if(ans.Length > 0){
                        fraction1 = ans;
                    }else{
                        fraction1 = expression.Substring(0,i);
                    }
                    ans = Calculate(fraction1 , fraction2 , expression[i]);
                    
                }
            }
        }
        if(ans.Length == 0){
            ans = expression;
        }
        return ans;
    }

    private string Calculate(string fraction1 , string fraction2 , char opera){
        string[] fra1nums = fraction1.Split('/');
        string[] fra2nums = fraction2.Split('/');
        int denominator = Convert.ToInt32(fra1nums[1]) *  Convert.ToInt32(fra2nums[1]);
        int factor1 = 1, factor2 = 1;
        if(fra1nums[0][0] == '-'){
            factor1 = -1;
            fra1nums[0] = fra1nums[0].Substring(1,fra1nums[0].Length - 1 );
        }
        if(fra2nums[0][0] == '-'){
            factor2 = -1;
            fra2nums[0] = fra2nums[0].Substring(1,fra2nums[0].Length - 1 );
        }
        
        int numerator1 = Convert.ToInt32(fra1nums[0])*factor1*Convert.ToInt32(fra2nums[1]);
        int numerator2 = Convert.ToInt32(fra1nums[1])*factor2*Convert.ToInt32(fra2nums[0]);
        int numerator = opera  == '+'? numerator1 + numerator2 :  numerator1 - numerator2;
        if(numerator == 0){
            return "0/1";
        }
        int gcd = this.GCD(Math.Abs(denominator),Math.Abs(numerator));
        numerator /= gcd;
        denominator /= gcd;
        if(numerator < 0){
            return '-'+Math.Abs(numerator).ToString()+'/' + denominator.ToString();
        }
        return numerator.ToString() + '/' + denominator.ToString();
    }

    private int GCD(int a, int b)
    {
        if (a % b == 0) return b;
        return GCD(b, a % b);
    }
}
```
***

## 2022/7/26

## 1206. 设计跳表

[原题地址 1206. 设计跳表](https://leetcode.cn/problems/design-skiplist/)

```
不使用任何库函数，设计一个 跳表 。

跳表 是在 O(log(n)) 时间内完成增加、删除、搜索操作的数据结构。跳表相比于树堆与红黑树，其功能与性能相当，并且跳表的代码长度相较下更短，其设计思想与链表相似。

例如，一个跳表包含 [30, 40, 50, 60, 70, 90] ，然后增加 80、45 到跳表中，以下图的方式操作：


Artyom Kalinin [CC BY-SA 3.0], via Wikimedia Commons

跳表中有很多层，每一层是一个短的链表。在第一层的作用下，增加、删除和搜索操作的时间复杂度不超过 O(n)。跳表的每一个操作的平均时间复杂度是 O(log(n))，空间复杂度是 O(n)。

了解更多 : https://en.wikipedia.org/wiki/Skip_list

在本题中，你的设计应该要包含这些函数：

bool search(int target) : 返回target是否存在于跳表中。
void add(int num): 插入一个元素到跳表。
bool erase(int num): 在跳表中删除一个值，如果 num 不存在，直接返回false. 如果存在多个 num ，删除其中任意一个即可。
注意，跳表中可能存在多个相同的值，你的代码需要处理这种情况。

 

示例 1:

输入
["Skiplist", "add", "add", "add", "search", "add", "search", "erase", "erase", "search"]
[[], [1], [2], [3], [0], [4], [1], [0], [1], [1]]
输出
[null, null, null, null, false, null, true, false, true, false]

解释
Skiplist skiplist = new Skiplist();
skiplist.add(1);
skiplist.add(2);
skiplist.add(3);
skiplist.search(0);   // 返回 false
skiplist.add(4);
skiplist.search(1);   // 返回 true
skiplist.erase(0);    // 返回 false，0 不在跳表中
skiplist.erase(1);    // 返回 true
skiplist.search(1);   // 返回 false，1 已被擦除

```

`思路`
关于[跳表 skiplist](https://github.com/h87545645/Blog/blob/main/data-structure/%E8%B7%B3%E8%A1%A8%20SkipList.md)

`c# 实现`
```
public class Skiplist {
    const int MAX_LEVEL = 10;
    const double P_FACTOR = 0.25;
    private SkiplistNode head;
    private int level;
    private Random random;
    public Skiplist() {
        this.head = new SkiplistNode(-1,MAX_LEVEL);
        this.level = 0;
        this.random = new Random();
    }
    
    public bool Search(int target) {
        
        
        SkiplistNode cur = this.head;
        //从最大level往下搜索路径
        for (int i = level - 1; i >= 0; i--)
        {
            //找到每层小于且最接近 target 的元素 
            while(cur.forward[i] != null && cur.forward[i].val < target){
                cur = cur.forward[i];
            }
        }
        cur = cur.forward[0];
        if (cur!= null && cur.val == target)
        {
            return true;
        }
        return false;
    }
    
    public void Add(int num) {
        //update 用于记录搜索路径中所有的SkiplistNode
        SkiplistNode[] update = new SkiplistNode[MAX_LEVEL];
        Array.Fill(update, head);
        SkiplistNode cur = this.head;
        //从最大level往下搜索路径
        for (int i = level - 1; i >= 0; i--)
        {
            while(cur.forward[i] != null && cur.forward[i].val < num){
                cur = cur.forward[i];
            }
            update[i] = cur;
        }
        int lv = this.GetRandom();
        this.level = Math.Max(lv,this.level);
        SkiplistNode newNode = new SkiplistNode(num,lv);
        for(int i =0;i<lv;i++){
            newNode.forward[i] = update[i].forward[i];
            update[i].forward[i] = newNode;
        }
    }
    
    public bool Erase(int num) {
        //update 用于记录搜索路径中所有的SkiplistNode
        SkiplistNode[] update = new SkiplistNode[MAX_LEVEL];
        SkiplistNode cur = head;
        //从最大level往下搜索路径
        for (int i = this.level - 1; i >= 0; i--)
        {
            while(cur.forward[i] != null && cur.forward[i].val < num){
                cur = cur.forward[i];
            }
            update[i] = cur;
        }
        cur = cur.forward[0];
        if(cur == null || cur.val != num){
            return false;
        }
        for(int i = 0;i< level; i++){
            if(update[i].forward[i] != cur){
               break; 
            }
            update[i].forward[i] = cur.forward[i];
        }
        //有为Null的层 需要更新level
        while(level > 1 && head.forward[level - 1] == null){
            level --;
        }
        return true;
    }
    
    private int GetRandom(){
        int lv = 1;
        while(random.NextDouble() < P_FACTOR && lv < MAX_LEVEL){
            lv ++;
        }
        return lv;
    }

    class SkiplistNode{
        public int val;
        public SkiplistNode[] forward;
        public SkiplistNode(int val,int maxLevel){
            this.val = val;
            this.forward = new SkiplistNode[maxLevel];
        }
    }
}
```

***

## 2022/7/25

## 919. 完全二叉树插入器

[原题地址 919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)

```
919. 完全二叉树插入器
完全二叉树 是每一层（除最后一层外）都是完全填充（即，节点数达到最大）的，并且所有的节点都尽可能地集中在左侧。

设计一种算法，将一个新节点插入到一个完整的二叉树中，并在插入后保持其完整。

实现 CBTInserter 类:

CBTInserter(TreeNode root) 使用头节点为 root 的给定树初始化该数据结构；
CBTInserter.insert(int v)  向树中插入一个值为 Node.val == val的新节点 TreeNode。使树保持完全二叉树的状态，并返回插入节点 TreeNode 的父节点的值；
CBTInserter.get_root() 将返回树的头节点。
 

示例 1：



输入
["CBTInserter", "insert", "insert", "get_root"]
[[[1, 2]], [3], [4], []]
输出
[null, 1, 2, [1, 2, 3, 4]]

解释
CBTInserter cBTInserter = new CBTInserter([1, 2]);
cBTInserter.insert(3);  // 返回 1
cBTInserter.insert(4);  // 返回 2
cBTInserter.get_root(); // 返回 [1, 2, 3, 4]
```

`思路`

这道题主要运用[二叉树的广度优先遍历BFS](https://github.com/h87545645/Blog/blob/main/algorithm/DFS%26BFS.md)，任意左或右节点为null的节点为待添加节点的父节点候选，定义Queue<TreeNode> nodeQueue 保存这些候选节点，Insert时将新节点作为队列首节点的左或右节点加入，加入后如果队列首节点左右都有节点了，则出队，并把新加的节点入队

`c# 实现`
```
public class CBTInserter {
    TreeNode myNode;
    Queue<TreeNode> nodeQueue;
    public CBTInserter(TreeNode root) {
        myNode = root;
        nodeQueue = new Queue<TreeNode>();
        Queue<TreeNode> temp = new Queue<TreeNode>();
        temp.Enqueue(myNode);
        while(temp.Count > 0){
            TreeNode nd = temp.Dequeue();
            if (nd.left == null || nd.right == null)
            {
                nodeQueue.Enqueue(nd);
            }
            if (nd.left != null)
            {
                 temp.Enqueue(nd.left);
            }
            if (nd.right != null)
            {
                 temp.Enqueue(nd.right);
            }
        }
    }
    
    public int Insert(int val) {
        TreeNode nd = nodeQueue.Peek();
        TreeNode child = new TreeNode(val);
        if (nd.left == null)
        {
            nd.left = child;
        }else{
            nd.right = child;
            nodeQueue.Dequeue();
        }
        nodeQueue.Enqueue(child);
        return nd.val;
    }
    
    public TreeNode Get_root() {
        return myNode;
    }
}
```

***

## 2022/7/22

## 757. 设置交集大小至少为2

[原题地址 757. 设置交集大小至少为2](https://leetcode.cn/problems/set-intersection-size-at-least-two/)

```
一个整数区间 [a, b]  ( a < b ) 代表着从 a 到 b 的所有连续整数，包括 a 和 b。

给你一组整数区间intervals，请找到一个最小的集合 S，使得 S 里的元素与区间intervals中的每一个整数区间都至少有2个元素相交。

输出这个最小集合S的大小。

示例 1:

输入: intervals = [[1, 3], [1, 4], [2, 5], [3, 5]]
输出: 3
解释:
考虑集合 S = {2, 3, 4}. S与intervals中的四个区间都有至少2个相交的元素。
且这是S最小的情况，故我们输出3。
示例 2:

输入: intervals = [[1, 2], [2, 3], [2, 4], [4, 5]]
输出: 5
解释:
最小的集合S = {1, 2, 3, 4, 5}.

```

`思路`
先对`intervals`按`intervals[0]`升序排序，如果相同按`intervals[1]`降序，定义 `List<int> areaTemp`记录所有区间的共同元素的集合，倒叙遍历`intervals`，按题目要求`intervals`中的区间至少要2个，如果`areaTemp`中有两个元素在`intervals[i]`区间中，则遍历下一个，否则将`intervals[i][0]`和`intervals[i][0]+1`加入`areaTemp` 

`c# 实现`

```
public class Solution {
    public int IntersectionSizeTwo(int[][] intervals) {
        Array.Sort(intervals,new CompareMethod());
        // return intervals[2][0];
        List<int> areaTemp = new List<int>();
        for (int i = intervals.Length - 1; i >= 0; i --)
        {
            int cnt = 2;
            if (areaTemp.Count > 0)
            {
                foreach (var item in areaTemp)
                {
                    if (item >= intervals[i][0] && item <= intervals[i][1])
                    {
                        cnt --;
                        if (cnt == 0)
                        {
                            break;
                        }
                    }
                
                }
              
            }
            if (cnt > 0)
            {
                for (int j = 0; j < cnt; j++)
                {
                   areaTemp.Add(intervals[i][0]+j);
                }
            }
        }
        // return areaTemp[1];
        return areaTemp.Count;
    }

    public class CompareMethod : IComparer<int[]>  //继承IComparer<T>接口，T为要比较的元素的类型
    {                                             //类中类，也可以放在类外面
        public  int Compare(int[] x, int[] y)
        {
            if (x[0] == y[0])
            {
                return y[1] - x[1];
            }else{ 
                return x[0] - y[0];
            }
        }
    }
}
```

***

## 2022/7/21

## 814. 二叉树剪枝

[原题地址 814. 二叉树剪枝](https://leetcode.cn/problems/binary-tree-pruning)

```
给你二叉树的根结点 root ，此外树的每个结点的值要么是 0 ，要么是 1 。

返回移除了所有不包含 1 的子树的原二叉树。

节点 node 的子树为 node 本身加上所有 node 的后代。

 

示例 1：


输入：root = [1,null,0,0,1]
输出：[1,null,0,null,1]
解释：
只有红色节点满足条件“所有不包含 1 的子树”。 右图为返回的答案。
示例 2：


输入：root = [1,0,1,0,0,0,1]
输出：[1,null,1,null,1]
示例 3：


输入：root = [1,1,0,1,1,0,1,0]
输出：[1,1,0,1,1,null,1]


```

`思路`

二叉树DFS应用，递归二叉树，节点为null或者左右节点都为null且`val`为0时，返回null,否则返回该节点

`c# 实现`

```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public TreeNode PruneTree(TreeNode root) {
        if (root == null)
        {
            return null;
        }
        root.left = PruneTree(root.left);
        root.right = PruneTree(root.right);
        if (root.left == null && root.right == null && root.val != 1)
        {
            return null;
        }
        return root;
    }
}
```

***

## 2022/7/20

## 1260. 二维网格迁移

[原题地址 1260. 二维网格迁移](https://leetcode.cn/problems/shift-2d-grid/)

```
给你一个 m 行 n 列的二维网格 grid 和一个整数 k。你需要将 grid 迁移 k 次。

每次「迁移」操作将会引发下述活动：

位于 grid[i][j] 的元素将会移动到 grid[i][j + 1]。
位于 grid[i][n - 1] 的元素将会移动到 grid[i + 1][0]。
位于 grid[m - 1][n - 1] 的元素将会移动到 grid[0][0]。
请你返回 k 次迁移操作后最终得到的 二维网格。

```
`思路`
题目要求其实就是将 `grid` 中每个元素向后移动 `k` 个位置，超过长度的循环到开头的位置。所以只需遍历一次`grid`，算出每个元素移动之后的下标位置，将计算好的下标位置的值添加到 `IList<IList<int>> ans`中，最后返回`ans`
`c# 实现`
```
public class Solution {
    public IList<IList<int>> ShiftGrid(int[][] grid, int k) {
    int m = grid.Length, n = grid[0].Length;
        IList<IList<int>> ans = new List<IList<int>>();
        // int[,] temp = new int[m,n];
        for(int i = 0; i < m; i++){
            IList<int> temp = new List<int>();
            for (int j = 0; j < n; j++)
            {
                // temp[i,j] = grid[i][j];
                int idx = i *  n + j;
                int preidx = (idx - k%(m*n) + n * m)%(m*n);
                int prei = (int)Math.Floor((double)(preidx / n));
                int prej = preidx % n;
                // grid[i][j] = grid[prei][prej];
                temp.Add(grid[prei][prej]);
            }
            ans.Add(temp);
        }
        return ans;
    }
}
```
做题中发现了c#的int[][] grid其实不是二维数组， [c# int [][] 和 int[,] 的区别](https://github.com/h87545645/Blog/blob/main/data-structure/c%23%20int%20%5B%5D%5B%5D%20%E5%92%8C%20int%5B%2C%5D%20%E7%9A%84%E5%8C%BA%E5%88%AB)
***

## 2022/7/19

## 731. 我的日程安排表 II

[原题地址 731. 我的日程安排表 II](https://leetcode.cn/problems/my-calendar-ii/)

相关问题 [729. 我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/)

```
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。

MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。

当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订。

每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致三重预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。

请按照以下步骤调用MyCalendar 类: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

 

示例：

MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(50, 60); // returns true
MyCalendar.book(10, 40); // returns true
MyCalendar.book(5, 15); // returns false
MyCalendar.book(5, 10); // returns true
MyCalendar.book(25, 55); // returns true
解释： 
前两个日程安排可以添加至日历中。 第三个日程安排会导致双重预订，但可以添加至日历中。
第四个日程安排活动（5,15）不能添加至日历中，因为它会导致三重预订。
第五个日程安排（5,10）可以添加至日历中，因为它未使用已经双重预订的时间10。
第六个日程安排（25,55）可以添加至日历中，因为时间 [25,40] 将和第三个日程安排双重预订；
时间 [40,50] 将单独预订，时间 [50,55）将和第二个日程安排双重预订。

```

`分析`
相比以前做过的的[我的日程安排表 I](https://leetcode.cn/problems/my-calendar-i/)，二增加了双重预定的要求，可以定义一个`calendar`存日程预定，定义另一个字典`repetCalendar`存预定了两次的部分，如果新book的日程与`repetCalendar`有重叠，则返回false,否则遍历`calendar`将重叠部分加入`repetCalendar`，接着更新`calendar`返回true

`c# 实现`

```
public class MyCalendarTwo {
    private Dictionary<int,int> calendar;
    private Dictionary<int,int> repetCalendar;
    public MyCalendarTwo() {
        calendar = new Dictionary<int,int>();
        repetCalendar = new Dictionary<int,int>();
    }
    
    public bool Book(int start, int end) {
        if(calendar.Count == 0){
            calendar.Add(start, end);
            return true;
        }
        if (repetCalendar.Count > 0)
        {   
            //判断是否已经重复预定了
            foreach( KeyValuePair<int, int> kvp in repetCalendar ){
                if(end > kvp.Key && start < kvp.Value ){
                    return false;
                }
            }
        }
        
        //与calendar比较，共同的部分加入repetCalendar 
        foreach( KeyValuePair<int, int> kvp in calendar ){
            if(end > kvp.Key && start < kvp.Value ){
                int tempend = Math.Min(kvp.Value,end);
                int tempstart = Math.Max(kvp.Key,start);
                repetCalendar.Add(tempstart , tempend);
            }
        }
        if(calendar.ContainsKey(start)){
            calendar[start] = Math.Max(calendar[start] , end);
        }else{
            calendar.Add(start, end);
        }
        return true;
    }

}
```


***
## 2022/7/18

## 749. 隔离病毒

[原题地址 749. 隔离病毒](https://leetcode.cn/problems/contain-virus/)

`分析`
遍历数组，遇到病毒将virusAreaCnt为key的VirusArea类计入virusAreaDict字典，并递归搜索周围的区域，遇0则VirusArea的哈希neighbors Add, wall计数加1，遇1则标记原数组为-virusAreaCnt。
遍历结束后如果virusAreaDict count <= 1 则返回结果，否则比较 找到neighbors最大的字典，将值设为1，其余字典还原为1 重复上述操作

`c# 实现`
```
public class Solution
{
    static int[][] dirs = {new int[]{-1, 0}, new int[]{1, 0}, new int[]{0, -1}, new int[]{0, 1}};
    private int virusAreaCnt ;
    private Dictionary<int, VirusArea> virusAreaDict;
    class VirusArea
    {
        public VirusArea(){}
        public HashSet<int> neighbors = new HashSet<int>();
        public int wall = 0;
    }

    public int ContainVirus(int[][] isInfected)
    {
        int ans = 0;
        int rowCnt = isInfected.Length, colCnt = isInfected[0].Length;
        while (true)
        {    
            virusAreaDict = new Dictionary<int, VirusArea>();
            virusAreaCnt = 0;
            for (int row = 0; row < rowCnt; row++)
            {
                for (int col = 0; col < colCnt; col++)
                {
                    if (isInfected[row][col] == 1)
                    {
                    
                        virusAreaCnt++;

                        if (!virusAreaDict.ContainsKey(virusAreaCnt))
                        {
                            virusAreaDict.Add(virusAreaCnt,new VirusArea());
      
                        }
                        Queue<Tuple<int, int>> queue = new Queue<Tuple<int, int>>();
                        queue.Enqueue(new Tuple<int, int>(row, col));
                        isInfected[row][col] = -virusAreaCnt;
                        while (queue.Count > 0) {
                            Tuple<int, int> tuple = queue.Dequeue();
                            int x = tuple.Item1, y = tuple.Item2;
                            for (int d = 0; d < 4; ++d) {
                                int nx = x + dirs[d][0], ny = y + dirs[d][1];
                                if (nx >= 0 && nx < rowCnt && ny >= 0 && ny < colCnt) {
                                    if (isInfected[nx][ny] == 1) {
                                        queue.Enqueue(new Tuple<int, int>(nx, ny));
                                        isInfected[nx][ny] = -virusAreaCnt;
                                    } else if (isInfected[nx][ny] == 0) {
                                        virusAreaDict[virusAreaCnt].wall++;
                                        int idx = nx * colCnt + (ny);
                                         virusAreaDict[virusAreaCnt].neighbors.Add(idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (virusAreaDict.Count == 0)
            {
                break;
            }
            int maxVirusAreaCnt = 0;
            int key = 1;
            foreach (KeyValuePair<int, VirusArea> kv in virusAreaDict)
            {
                if(maxVirusAreaCnt < kv.Value.neighbors.Count){
                    maxVirusAreaCnt = kv.Value.neighbors.Count;
                    key = kv.Key;
                }
                maxVirusAreaCnt = Math.Max(maxVirusAreaCnt,kv.Value.neighbors.Count);
            }
            ans += virusAreaDict[key].wall;
            if (virusAreaDict.Count == 1)
            {
                break;
            }
            for (int row = 0; row < rowCnt; row++)
            {
                for (int col = 0; col < colCnt; col++)
                {
                    if (isInfected[row][col] < 0)
                    {
                        if (isInfected[row][col] != -key)
                        {
                            isInfected[row][col] = 1;
                        }
                        else
                        {
                            isInfected[row][col] = 2;
                        }
                    }
                }
            }
            foreach (KeyValuePair<int, VirusArea> kv in virusAreaDict)
            {
                if (kv.Key != key)
                {
                    int cnt = 0;
                    foreach (int idx in kv.Value.neighbors)
                    {
                        int row = (int)Math.Floor((double)(idx / colCnt));
                        int col = idx % colCnt;
                      
                        isInfected[row][col] = 1;
                        cnt ++;
                    }
                }
            }
        }
        
        return ans;
    }

}
```

其中用到的[元组 Tuple](https://github.com/h87545645/Blog/blob/main/data-structure/c%23%20%E5%85%83%E7%BB%84%20Tuple.md)

***

## 2022/7/15

## 558. 四叉树交集

[原题地址 558. 四叉树交集](https://leetcode.cn/problems/logical-or-of-two-binary-grids-represented-as-quad-trees/)

`题目比较长就不复制了，这道题主要难在四叉树的理解，实现一个递归方法，比较quadTree1和quadTree2，只要其中有一个是叶子isLeaf，则根据条件返回其中一个node。如果两个都不是叶子，则递归四个字节点合成一个节点，新节点根据规则判断是否为叶子`

`c# 实现`
```
public class Solution {

    public Node Intersect(Node quadTree1, Node quadTree2) {
        
        if (quadTree1.isLeaf || quadTree2.isLeaf)
        {
             if (quadTree1.isLeaf && quadTree1.val)
            {
                return new Node(quadTree1.val,quadTree1.isLeaf);
            }else if(quadTree1.isLeaf && !quadTree1.val){
                return new Node(quadTree2.val,quadTree2.isLeaf,quadTree2.topLeft,quadTree2.topRight,quadTree2.bottomLeft,quadTree2.bottomRight);
            }
            else if (quadTree2.isLeaf && quadTree2.val)
            {
                return new Node(quadTree2.val,quadTree2.isLeaf);
            }else{
                return new Node(quadTree1.val,quadTree1.isLeaf,quadTree1.topLeft,quadTree1.topRight,quadTree1.bottomLeft,quadTree1.bottomRight);
            }
        }else
        {
            Node o1 = Intersect(quadTree1.topLeft, quadTree2.topLeft);
        Node o2 = Intersect(quadTree1.topRight, quadTree2.topRight);
        Node o3 = Intersect(quadTree1.bottomLeft, quadTree2.bottomLeft);
        Node o4 = Intersect(quadTree1.bottomRight, quadTree2.bottomRight);
        if (o1.isLeaf && o2.isLeaf && o3.isLeaf && o4.isLeaf && o1.val == o2.val && o1.val == o3.val && o1.val == o4.val) {
            return new Node(o1.val, true);
        }
        return new Node(true, false, o1, o2, o3, o4);
        }
    }
}
```

***

## 2022/7/14

## 745. 前缀和后缀搜索

[原题地址 745. 前缀和后缀搜索](https://leetcode.cn/problems/prefix-and-suffix-search/)

```
设计一个包含一些单词的特殊词典，并能够通过前缀和后缀来检索单词。

实现 WordFilter 类：

WordFilter(string[] words) 使用词典中的单词 words 初始化对象。
f(string pref, string suff) 返回词典中具有前缀 prefix 和后缀 suff 的单词的下标。如果存在不止一个满足要求的下标，返回其中 最大的下标 。如果不存在这样的单词，返回 -1 。
 

示例：

输入
["WordFilter", "f"]
[[["apple"]], ["a", "e"]]
输出
[null, 0]
解释
WordFilter wordFilter = new WordFilter(["apple"]);
wordFilter.f("a", "e"); // 返回 0 ，因为下标为 0 的单词：前缀 prefix = "a" 且 后缀 suff = "e" 。


```

`分析`

本来想用每个单词的前后缀拼接作为key来构造[字典树 Trie](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)的。后来看到官方题解里有更简洁的方法。

```
计算每个单词的前缀后缀组合可能性

预先计算出每个单词的前缀后缀组合可能性，用特殊符号连接，作为键，对应的最大下标作为值保存入哈希表。检索时，同样用特殊符号连接前后缀，在哈希表中进行搜索。

```

`c#实现`

```
public class WordFilter {
    Dictionary<string, int> dictionary;

    public WordFilter(string[] words) {
        dictionary = new Dictionary<string, int>();
        for (int i = words.Length - 1; i >= 0; i--) {
            string word = words[i];
            int m = word.Length;
            for (int prefixLength = 1; prefixLength <= m; prefixLength++) {
                for (int suffixLength = 1; suffixLength <= m; suffixLength++) {
                    dictionary.TryAdd(word.Substring(0, prefixLength) + "#" + word.Substring(m - suffixLength), i);
                }
            }
        }
    }

    public int F(string pref, string suff) {
        if (dictionary.ContainsKey(pref + "#" + suff)) {
            return dictionary[pref + "#" + suff];
        }
        return -1;
    }
}

```

***

## 2022/7/13
## 735. 行星碰撞

[原题地址 735. 行星碰撞](https://leetcode.cn/problems/asteroid-collision/)

```
给定一个整数数组 asteroids，表示在同一行的行星。

对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。

找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。

 

示例 1：

输入：asteroids = [5,10,-5]
输出：[5,10]
解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
示例 2：

输入：asteroids = [8,-8]
输出：[]
解释：8 和 -8 碰撞后，两者都发生爆炸。
示例 3：

输入：asteroids = [10,2,-5]
输出：[10]
解释：2 和 -5 发生碰撞后剩下 -5 。10 和 -5 发生碰撞后剩下 10 。

```

`分析`

遍历asteroids，int i 指向头，int j 指向尾。每次循环 i j 下标的行星分别判断是否和相邻的相撞，相撞时删除对应行星。

`c# 实现`

```
public class Solution {
    public int[] AsteroidCollision(int[] asteroids) {
        List<int> aster = new List<int>(asteroids);
         for (int i = 0 , j = asteroids.Length - 1; i < asteroids.Length || j >= 0; i++ , j --)
            {
                if (i > 0 && i < asteroids.Length && asteroids[i] < 0 && asteroids[i - 1] > 0 )
                {
                    int sub = asteroids[i - 1] + asteroids[i];
                    // return new int[]{sub};
                    if (sub == 0)
                    {
                        asteroids = DeleteEle(asteroids , i - 1 , 2);
                        i -= 2;
                    }else if (sub > 0)
                    {
                        asteroids = DeleteEle(asteroids , i , 1);
                        i -= 1;
                    }else{
                        asteroids = DeleteEle(asteroids , i - 1 , 1);
                        i -= 1;
                    }
                }
                if (j < asteroids.Length - 1 && i >=0 && asteroids[j] > 0 && asteroids[j + 1] < 0 )
                {
                    int sub = asteroids[j + 1] + asteroids[j];
                    // return new int[]{sub};
                    if (sub == 0)
                    {
                        asteroids = DeleteEle(asteroids , j  , 2);
                        j += 2;
                    }else if (sub > 0)
                    {
                        asteroids = DeleteEle(asteroids , j + 1 , 1);
                        j += 1;
                    }else{
                        asteroids = DeleteEle(asteroids , j  , 1);
                        j += 1;
                    }
                }
            }
        return asteroids;
    }

    private int[] DeleteEle(int[] arrayBorn,int index,int Len) 
    {
        if (Len < 0)    //删除长度小于0，返回
        {
            return arrayBorn;
        }
        if ( (index + Len) > arrayBorn.Length)      //删除长度超出数组范围 
        {
            Len = arrayBorn.Length - index;         //将长度设置为能删除的最大长度
        }
        for (int i = 0;i < arrayBorn.Length - ( index + Len); i++)      //将删除元素后面的元素往前移动
        {
            if ((index + i + Len) > arrayBorn.Length)       //若删除元素+删除长度超过数组的范围，即无法从数组中找到移动元素，则用null替代
            {
                arrayBorn[index + i] = 0;
            }
            else            //若能用数组的元素替换则用数组中的元素
            {
                arrayBorn[index + i] = arrayBorn[index + i + Len];
            }           
        }
        /*不改变数组长度*/
        // for (int j =Len;j > 0; j--)         //将删除元素后多余的元素置为null值
        // {
        //     arrayBorn[arrayBorn.Length - j ] = null;
        // }
        int[] newArray = new int[arrayBorn.Length-Len];
        for (int j =0;j < newArray.Length;j++) 
        {
            newArray[j] = arrayBorn[j];
        }
        return newArray;
        // return arrayBorn;            
    }
}

```


***

## 2022/7/12
## 1252. 奇数值单元格的数目

```
给你一个 m x n 的矩阵，最开始的时候，每个单元格中的值都是 0。

另有一个二维索引数组 indices，indices[i] = [ri, ci] 指向矩阵中的某个位置，其中 ri 和 ci 分别表示指定的行和列（从 0 开始编号）。

对 indices[i] 所指向的每个位置，应同时执行下述增量操作：

ri 行上的所有单元格，加 1 。
ci 列上的所有单元格，加 1 。
给你 m、n 和 indices 。请你在执行完所有 indices 指定的增量操作后，返回矩阵中 奇数值单元格 的数目。

 

示例 1：



输入：m = 2, n = 3, indices = [[0,1],[1,1]]
输出：6
解释：最开始的矩阵是 [[0,0,0],[0,0,0]]。
第一次增量操作后得到 [[1,2,1],[0,1,0]]。
最后的矩阵是 [[1,3,1],[1,3,1]]，里面有 6 个奇数。
示例 2：



输入：m = 2, n = 2, indices = [[1,1],[0,0]]
输出：0
解释：最后的矩阵是 [[2,2],[2,2]]，里面没有奇数。
 

提示：

1 <= m, n <= 50
1 <= indices.length <= 100
0 <= ri < m
0 <= ci < n
 

进阶：你可以设计一个时间复杂度为 O(n + m + indices.length) 且仅用 O(n + m) 额外空间的算法来解决此问题吗？

```

[原题地址 1252. 奇数值单元格的数目](https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix)

`分析`

根据题目要求，先遍历indices可以用两个数组分别记录行列增加的次数，再遍历矩阵，计算每个位置行列相加并且计数奇数。

`c#实现`

```
public class Solution {
    public int OddCells(int m, int n, int[][] indices) {
        int[] r = new int[m];
        int[] c = new int[n];
        int ans = 0;
       
        for(int i =0; i < indices.Length; i ++){
            r[indices[i][0]] ++;
            c[indices[i][1]] ++;
        }
        
        for (int i = 0; i < r.Length; i++)
        {
            for (int j = 0; j < c.Length; j++)
            {
                if (((r[i] + c[j]) & 1) == 1)
                {
                    ans ++;
                }
            }
        }
        return ans;
    }
}
```
关于[& 运算符](https://github.com/h87545645/Blog/blob/main/c%23/c%23%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93.md#-%E8%BF%90%E7%AE%97%E7%AC%A6)

***

## 2022/7/11
## 676. 实现一个魔法字典
```
设计一个使用单词列表进行初始化的数据结构，单词列表中的单词 互不相同 。 如果给出一个单词，请判定能否只将这个单词中一个字母换成另一个字母，使得所形成的新单词存在于你构建的字典中。

实现 MagicDictionary 类：

MagicDictionary() 初始化对象
void buildDict(String[] dictionary) 使用字符串数组 dictionary 设定该数据结构，dictionary 中的字符串互不相同
bool search(String searchWord) 给定一个字符串 searchWord ，判定能否只将字符串中 一个 字母换成另一个字母，使得所形成的新字符串能够与字典中的任一字符串匹配。如果可以，返回 true ；否则，返回 false 。
示例：

输入
["MagicDictionary", "buildDict", "search", "search", "search", "search"]
[[], [["hello", "leetcode"]], ["hello"], ["hhllo"], ["hell"], ["leetcoded"]]
输出
[null, null, false, true, false, false]

解释
MagicDictionary magicDictionary = new MagicDictionary();
magicDictionary.buildDict(["hello", "leetcode"]);
magicDictionary.search("hello"); // 返回 False
magicDictionary.search("hhllo"); // 将第二个 'h' 替换为 'e' 可以匹配 "hello" ，所以返回 True
magicDictionary.search("hell"); // 返回 False
magicDictionary.search("leetcoded"); // 返回 False
```

[原题地址 676. 实现一个魔法字典](https://leetcode.cn/problems/implement-magic-dictionary/)

`分析`

读完题就感觉适合使用上周用过的 **[字典树 Trie](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)**  来实现。

在buildDict中添加字典树
```
class Trie{
    public bool IsFinished { get; set; }
    public Trie[] Child { get; set; }
    public Trie(){
        IsFinished = false;
        Child = new Trie[26];
    }
}
public void BuildDict(string[] dictionary) {
    foreach (string word in dictionary)
    {
        Trie cur = root;
        for (int i = 0; i < word.Length; i++)
        {
            int index = word[i] - 'a';
            if (cur.Child[index] == null)
            {
                cur.Child[index] = new Trie();
            }
            cur = cur.Child[index];
        }
        cur.IsFinished = true;
    }
}
```

Search里递归DFS方法，如果字典树中第一次找不到此字符，则用isModify标记修改了单词，第二次找不到单词或这在index == searchWord.Length时未被isModify，则返回false

```
 private bool DFS(string searchWord , int index ,Trie node, bool isModify){
    if (index == searchWord.Length)
    {
        return isModify && node.IsFinished;
    }
    int wordInex = searchWord[index] - 'a';
    if (node.Child[wordInex] != null)
    {
        if (DFS(searchWord , index + 1 , node.Child[wordInex] , isModify))
        {
             return true;
        }
    }
    if (!isModify)
    {
        for (int i = 0; i < 26; i++)
        {
            if (i != wordInex && node.Child[i] != null)
            {
                if (DFS(searchWord , index + 1 , node.Child[i] , true))
                {
                    return true;
                }
            }
        }
    }
    return false;
}
```

***

## 2022/7/8
## 1217. 玩筹码

>有 n 个筹码。第 i 个筹码的位置是 position[i] 。

>我们需要把所有筹码移到同一个位置。在一步中，我们可以将第 i 个筹码的位置从 position[i] 改变为:

>position[i] + 2 或 position[i] - 2 ，此时 cost = 0
>position[i] + 1 或 position[i] - 1 ，此时 cost = 1
>返回将所有筹码移动到同一位置上所需要的 最小代价 。

[原题地址 1217. 玩筹码](https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/)

`分析`

按照一般思路需要遍历postion,算出其他筹码移动到该下标的cost，最后取cost最小的一个。但这种方式要遍历两次position，时间复杂度为O(n的2次方)。此题多半不是考虑这种解法。观察cost规则，只有移动奇数位cost才会+1，可以用[贪心算法](https://github.com/h87545645/Blog/blob/main/algorithm/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95.md)把偶数和奇数位筹码看作整体，结果只和最终位置是偶数位还是奇数位相关，所以只需遍历一次positon，记录奇(偶)数位置的个数，取小的返回就是最终结果。

`c# 实现`
```
public class Solution {
    public int MinCostToMoveChips(int[] position) {
        int even = 0, odd = 0;
        foreach (int pos in position) {
            if ((pos & 1) != 0) { //判断奇偶
                odd++;
            } else {
                even++;
            }
        }
        return Math.Min(odd, even);
    }
}

```

关于[& 运算符](https://github.com/h87545645/Blog/blob/main/c%23/c%23%E7%BB%8F%E9%AA%8C%E6%80%BB%E7%BB%93.md#-%E8%BF%90%E7%AE%97%E7%AC%A6)
​
 
***

## 2022/7/7
## 648. 单词替换

在英语中，我们有一个叫做 词根(root) 的概念，可以词根后面添加其他一些词组成另一个较长的单词——我们称这个词为 继承词(successor)。例如，词根an，跟随着单词 other(其他)，可以形成新的单词 another(另一个)。

现在，给定一个由许多词根组成的词典 dictionary 和一个用空格分隔单词形成的句子 sentence。你需要将句子中的所有继承词用词根替换掉。如果继承词有许多可以形成它的词根，则用最短的词根替换它。

你需要输出替换之后的句子。

```
示例 1：

输入：dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
输出："the cat was rat by the bat"
示例 2：

输入：dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
输出："a a b c"
 

提示：

1 <= dictionary.length <= 1000
1 <= dictionary[i].length <= 100
dictionary[i] 仅由小写字母组成。
1 <= sentence.length <= 10^6
sentence 仅由小写字母和空格组成。
sentence 中单词的总量在范围 [1, 1000] 内。
sentence 中每个单词的长度在范围 [1, 1000] 内。
sentence 中单词之间由一个空格隔开。
sentence 没有前导或尾随空格。
```
`分析`

**由于要求用最短的词根替换，考虑对于sentence中的每个单词，由短至长遍历它所有的前缀。把dictionary 中所有词根放入哈希集合中，如果这个前缀出现在哈希集合中，则我们找到了当前单词的最短词根，将这个词根替换原来的单词。最后返回重新拼接的句子**

`c#实现`
```
public class Solution {
    public string ReplaceWords(IList<string> dictionary, string sentence) {
        ISet<string> dictionarySet = new HashSet<string>();
        foreach (string root in dictionary) {
            dictionarySet.Add(root);
        }
        string[] words = sentence.Split(" ");
        for (int i = 0; i < words.Length; i++) {
            string word = words[i];
            for (int j = 0; j < word.Length; j++) {
                if (dictionarySet.Contains(word.Substring(0, 1 + j))) {
                    words[i] = word.Substring(0, 1 + j);
                    break;
                }
            }
        }
        return String.Join(" ", words);
    }
}

```

`解法二 字典树`

**看到一种有意思的解法，`字典树` 关于字典树 [字典树](https://github.com/h87545645/Blog/blob/main/data-structure/%E5%AD%97%E5%85%B8%E6%A0%91.md)**

**我们用 dictionary 中所有词根构建一棵字典树，并用特殊符号标记结尾。在搜索前缀时，只需在字典树上搜索出一条最短的前缀路径即可。**

`c#实现`
```
public class Solution {
    public string ReplaceWords(IList<string> dictionary, string sentence) {
        Trie trie = new Trie();
        foreach (string word in dictionary) {
            Trie cur = trie;
            for (int i = 0; i < word.Length; i++) {
                char c = word[i];
                if (!cur.Children.ContainsKey(c)) {
                    cur.Children.Add(c, new Trie());
                }
                cur = cur.Children[c];
            }
            cur.Children.Add('#', new Trie());
        }
        string[] words = sentence.Split(" ");
        for (int i = 0; i < words.Length; i++) {
            words[i] = FindRoot(words[i], trie);
        }
        return string.Join(" ", words);
    }

    public string FindRoot(string word, Trie trie) {
        StringBuilder root = new StringBuilder();
        Trie cur = trie;
        for (int i = 0; i < word.Length; i++) {
            char c = word[i];
            if (cur.Children.ContainsKey('#')) {
                return root.ToString();
            }
            if (!cur.Children.ContainsKey(c)) {
                return word;
            }
            root.Append(c);
            cur = cur.Children[c];
        }
        return root.ToString();
    }
}

public class Trie {
    public Dictionary<char, Trie> Children;

    public Trie() {
        Children = new Dictionary<char, Trie>();
    }
}

```

***

## 2022/7/6
## 736. Lisp 语法解析
给你一个类似 Lisp 语句的字符串表达式 expression，求出其计算结果。

表达式语法如下所示:

表达式可以为整数，let 表达式，add 表达式，mult 表达式，或赋值的变量。表达式的结果总是一个整数。
(整数可以是正整数、负整数、0)
let 表达式采用 "(let v1 e1 v2 e2 ... vn en expr)" 的形式，其中 let 总是以字符串 "let"来表示，接下来会跟随一对或多对交替的变量和表达式，也就是说，第一个变量 v1被分配为表达式 e1 的值，第二个变量 v2 被分配为表达式 e2 的值，依次类推；最终 let 表达式的值为 expr表达式的值。
add 表达式表示为 "(add e1 e2)" ，其中 add 总是以字符串 "add" 来表示，该表达式总是包含两个表达式 e1、e2 ，最终结果是 e1 表达式的值与 e2 表达式的值之 和 。
mult 表达式表示为 "(mult e1 e2)" ，其中 mult 总是以字符串 "mult" 表示，该表达式总是包含两个表达式 e1、e2，最终结果是 e1 表达式的值与 e2 表达式的值之 积 。
在该题目中，变量名以小写字符开始，之后跟随 0 个或多个小写字符或数字。为了方便，"add" ，"let" ，"mult" 会被定义为 "关键字" ，不会用作变量名。
最后，要说一下作用域的概念。计算变量名所对应的表达式时，在计算上下文中，首先检查最内层作用域（按括号计），然后按顺序依次检查外部作用域。测试用例中每一个表达式都是合法的。有关作用域的更多详细信息，请参阅示例。


`示例：`
```
示例 1：

输入：expression = "(let x 2 (mult x (let x 3 y 4 (add x y))))"
输出：14
解释：
计算表达式 (add x y), 在检查变量 x 值时，
在变量的上下文中由最内层作用域依次向外检查。
首先找到 x = 3, 所以此处的 x 值是 3 。
示例 2：

输入：expression = "(let x 3 x 2 x)"
输出：2
解释：let 语句中的赋值运算按顺序处理即可。
示例 3：

输入：expression = "(let x 1 y 2 x (add x y) (add x y))"
输出：5
解释：
第一个 (add x y) 计算结果是 3，并且将此值赋给了 x 。 
第二个 (add x y) 计算结果是 3 + 2 = 5 。


```
`分析`
***每个表达式都包含在()里，且有let add mult 三种关键字，其余都是变量名和整数，可以实现一个EvaInt函数来返回当前下标得值并移动下标，实现一个EvaVar函数来返回当前下标的变量字符穿并移动下标，因为let的赋值可能有多个，变量需要在对应作用域下，可以申明一个Dictionary<string,Stack<int>> scope 来存对应的变量值。考虑实现一个递归函数InnerEva
返回表达式的结果，只要当前字符不为左括号’(’，则判断为变量或值，直接返回变量或整数，然后判断判断是let add 或 mult 。add返回两变量的和，mult返回两变量的积， let 需要赋值所有的变量直到遇到左右括号（）则递归（）里的表达式。***

`c#实现`
```
public class Solution {
    int index = 0;
    Dictionary<string,Stack<int>> scope = new Dictionary<string,Stack<int>>(); //用来记录作用域内所有变量的值
    public int Evaluate(string expression) {
        return InnerEva(expression);
    }
    
    private int InnerEva(string expression){
        //不是（ 则只可能是 变量或者值 返回值或者变量的值
        if (expression[index] != '(')
        {
            if (char.IsLower(expression[index]))
            {
                string var = EvaVar(expression);
                return scope[var].Peek();
            }else{ //整数
                return EvaInt(expression);
            }   
        }
        // 下面处理括号内的表达式
        //移除(
        int ret;
        index++;
        //判断是let add 或 mult
        if (expression[index] == 'l')
        {
            index += 4;
            IList<string> vars = new List<string>(); //记录所有的变量名
            while (true)
            {
                if (!char.IsLower(expression[index])) //如果不是变量的字符 这时应该是下一个表达式的（  此时直接递得到下一个（）的值
                {
                    ret = InnerEva(expression);
                    break;
                }
                //记录该scope let表达式里 所有变量的值
                string var = EvaVar(expression);
                if (expression[index] == ')') //如果let 表达式结束 则返回最后赋值的变量值
                {
                    ret = scope[var].Peek();
                    break;
                }
                vars.Add(var);
                index ++;
                int v = InnerEva(expression);
                if (!scope.ContainsKey(var))
                {
                    scope.Add(var,new Stack<int>());
                }
                scope[var].Push(v);
                index++;
            }
            foreach (string var in vars) {
                scope[var].Pop(); // 清除当前作用域的变量
            }
        }else if (expression[index] == 'a')
        {
            index += 4;
            int v1 = InnerEva(expression);
            index ++;
            int v2 = InnerEva(expression);
            ret = v1 + v2;
        }else
        {
            index += 5;
            int v1 = InnerEva(expression);
            index ++;
            int v2 = InnerEva(expression);
            ret = v1 * v2;
        }
        //移除 ）
        index ++;
        return ret;
    }

    //返回当前下表的值
    private int EvaInt(string expression){
        int n = expression.Length;
        int ret = 0, sign = 1;
        if(expression[index] == '-'){
            sign = -1;
            index ++;
        }
        while (index < n && char.IsDigit(expression[index]))
        {
            ret = ret * 10 + (expression[index] - '0');
            index ++;
        }
        return ret*sign;
    }

    //返回当前下表的变量字符
    private string EvaVar(string expression){
        int n = expression.Length;
        StringBuilder ret = new StringBuilder();
        while (index < n && expression[index] != ' ' && expression[index] != ')')
        {
            ret.Append(expression[index]);
            index ++;
        }
        return ret.ToString();
    }
}
```


***


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
