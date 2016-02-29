require 'nn'

function criterionJacobianTest1DTable(cri, input0, target)
   -- supposes input is a tensor, which is splitted in the first dimension
   local precision = 1e-5
   local input = input0:split(1,1)
   for i=1,#input do
      input[i] = input[i][1]
   end
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(input0)
   local input_s = input0:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input0:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end
   local centraldiff_dfdx_t = centraldiff_dfdx:split(1,1)
   for i=1,#centraldiff_dfdx_t do
      centraldiff_dfdx_t[i] = centraldiff_dfdx_t[i][1]
   end
   for i=1,#centraldiff_dfdx_t do
      -- compare centraldiff_dfdx with :backward()
      local err = (centraldiff_dfdx_t[i] - dfdx[i]):abs():max()
      print(string.format('%s test: %g error in difference between central difference and :backward', torch.type(cri), err))
   end
end

