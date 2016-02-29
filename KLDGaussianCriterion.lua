--
--
--  KLDGaussianCriterion.lua
--  Created by Andrey Kolishchak on 2/14/16.
--
--
-- KL divergence of Gaussian Criterion
-- input = {mu, log(sigma^2)} = {mu, log_sq_sigma}
-- L = -0.5*(1 + log_sq_sigma - mu^2 - exp(log_sq_sigma)) 
--
--

require 'nn'
require 'test'

local KLDGaussianCriterion, parent = torch.class('nn.KLDGaussianCriterion', 'nn.Criterion')

function KLDGaussianCriterion:__init()
  parent.__init(self)
  
  self.gradInput = { torch.Tensor(), torch.Tensor() }
end

function KLDGaussianCriterion:updateOutput(input, target)
  local mu, log_sq_sigma = input[1], input[2]
  
  local output = torch.exp(log_sq_sigma):add(torch.pow(mu,2)):add(-1, log_sq_sigma):add(-1):mul(0.5):div(mu:size(1))
  
  self.output = torch.sum(output)
  
  return self.output
end

function KLDGaussianCriterion:updateGradInput(input, target)
  local mu, log_sq_sigma = input[1], input[2]
  
  -- d_L/d_mu = mu
  self.gradInput[1]:resizeAs(mu):copy(mu):div(mu:size(1))
  
  -- d_L/d_sigma = 0.5*(exp(log_sq_sigma)-1)
  self.gradInput[2] = torch.exp(log_sq_sigma):add(-1):mul(0.5):div(log_sq_sigma:size(1))
  
  return self.gradInput
  
end

function test()
  local model = nn.KLDGaussianCriterion()
  
  local input = torch.rand(2, 100)
  local target = torch.rand(100)
  
  local precision = 1e-5
  criterionJacobianTest1DTable(model, input, target)
end

test()
