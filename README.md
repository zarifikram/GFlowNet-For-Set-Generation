# GFlowNet For Set Generation
 Generation of set with duplicate values using trajectory balance.

# Example 
 `num_actions` = 10
 `size` = 4
 Possible generated set = `[3, 4, 1, 9]`

# Action Space
Total action space = `num_actions` * 2 + 1
Actions = {`left_append_actions`, `right_append_actions`, `terminate`}

Here `left_append_actions` push a number to the left, `right_append_actions` push a number to the right.
