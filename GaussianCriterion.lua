--
--
--  GaussianCriterion.lua
--  Created by Andrey Kolishchak on 2/14/16.
--
--
-- Gaussian log-likelihood Criterion
-- input = {mu, log(sigma^2)} = {mu, log_sq_sigma}
-- L = log(p(N(x,mu,sigma)) = 0.5*log_sq_sigma +0.5*log(2*pi) + 0.5*((x-mu)^2)/exp(log_sq_sigma)
--
--

require 'nn'
require 'test'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:__init()
  parent.__init(self)
  
  self.gradInput = { torch.Tensor(), torch.Tensor() }
end

function GaussianCriterion:updateOutput(input, target)
  local mu, log_sq_sigma = input[1], input[2]

  local sq_sigma = torch.exp(log_sq_sigma) -- sigma^2
  local output = torch.add(target, -1, mu):pow(2):cdiv(sq_sigma) -- ((x-mu)^2)/(sigma^2)
  output:add(log_sq_sigma) -- log(sigma^2)
  output:add(math.log(2*math.pi)) -- log(2*pi)
  output:mul(0.5):div(mu:size(1))

  self.output = torch.sum(output)
  
  return self.output
end

function GaussianCriterion:updateGradInput(input, target)
  local mu, log_sq_sigma = input[1], input[2]
  
  local sq_sigma = torch.exp(log_sq_sigma)
  
  -- d_L/d_mu = -(x-mu)/exp(log_sq_sima)
  local diff = torch.add(target, -1, mu)
  self.gradInput[1] = torch.cdiv(diff, sq_sigma):mul(-1):div(mu:size(1))
  
  -- d_L/d_sigma = 0.5-0.5*(x-mu)^2/exp(log_sq_sima)
  self.gradInput[2] = torch.pow(diff,2):mul(-0.5):cdiv(sq_sigma):add(0.5):div(log_sq_sigma:size(1))
  
  return self.gradInput
end

function test()
  local model = nn.GaussianCriterion()
  
  local input = torch.rand(2, 100)
  local target = torch.rand(100)
    
  local precision = 1e-5
  criterionJacobianTest1DTable(model, input, target)
end

test()
