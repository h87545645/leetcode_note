# LEET CODE STUDY NOTE

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
很巧妙的一道题，思路是将n转换成3进制来表达，比如12的3进制位 (110) 也就是12可以由 一个3的2次方 + 一个3的一次方 + 0 = 12 来构成，我们把n转换成3进制，每一位上只要是0或者1即可满足题目#不同#的三的幂之和的要求，如果有2则说明有重复的次幂则不满足要求，实现时我们不断对n除3并取余，除3的目的是使n右移（类比二进制的>>操作）来判断每一位上是否有2.

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
