--
--
--  Gaussian.lua
--  Created by Andrey Kolishchak on 2/14/16.
--
--
-- Gaussian sampler Module
-- input = {mu, log(sigma^2)} = {mu, log_sq_sigma}
-- y = mu + sqrt(exp(log_sq_sigma))*z = mu + exp(0.5*log_sq_sigma))*z, z ~ N(0,1)
--
local Gaussian, parent = torch.class('nn.Gaussian', 'nn.Module')

function Gaussian:__init()
  parent.__init(self)
  
  self.z = torch.Tensor()
  self.gradInput = { torch.Tensor(), torch.Tensor() }
end
 
 
function Gaussian:updateOutput(input)  
  local mu, log_sq_sigma = input[1], input[2]

  self.z:resizeAs(mu):normal(0, 1) -- z ~ N(0,1)
  self.output:resizeAs(log_sq_sigma):copy(log_sq_sigma):mul(0.5):exp():cmul(self.z):add(mu) -- exp(0.5*log_sq_sigma)*z + mu
  
  return self.output
end

function Gaussian:updateGradInput(input, gradOutput)
  local mu, log_sq_sigma = input[1], input[2]
  
  -- d_y/d_mu = 1
  self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
  
  -- d_y/d_sigma = exp(0.5*log_sq_sigma)*z*0.5
  self.gradInput[2]:resizeAs(log_sq_sigma):copy(log_sq_sigma):mul(0.5):exp():cmul(self.z):mul(0.5):cmul(gradOutput)
  
  return self.gradInput
end
