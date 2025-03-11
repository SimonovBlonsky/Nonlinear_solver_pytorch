# This file aims to construct a linear problem of visual-inertial bundle adjustment.
import torch
from enum import Enum

from BaseState import Vertex
from BaseFactor import NonlinearFactor

class Problem:
    def __init__(self):
        self.problem_type = "SLAM_PROBLEM"
        
        # Linear equation
        self.Hessian_ = None
        self.b_ = None
        self.delta_x_ = None
        
        self.currentLambda_ = 0.0
        self.currentChi_ = 0.0
        self.stopThresholdLM_ = 0.0    # Condition for LM iterations
        self.ni_ = 0.0                 # 控制 Lambda 缩放大小
        
        # Prior info
        self.H_prior_ = None
        self.b_prior_ = None
        self.b_prior_backup_ = None
        self.err_prior_backup_ = None

        self.Jt_prior_inv_ = None
        self.err_prior_ = None

        # Schur BA Pose
        self.H_pp_schur_ = None
        self.b_pp_schur_ = None
        
        # Hessian's Landmark and pose
        self.H_pp_ = None
        self.b_pp_ = None
        self.H_ll_ = None
        self.b_ll_ = None
        
        # Vertices and edges (States and factors)
        self.vertices_ = {}
        self.edges_ = {}
        
        # Ordering related
        self.ordering_poses_ = 0
        self.ordering_landmarks_ = 0
        self.ordering_generic_ = 0
        self.idx_pose_vertices_ = {}
        self.idx_landmark_vertices_ = {}
        
        # Vertices need to marg
        self.vertices_marg_ = None
        
    def Solve(self, iterations: int):
        
        # Get optimization dimensions
        self.SetOrdering()
        
        # From a big Hessian matrix
        self.MakeHessian()
        
        # LM initialization
        self.ComputeLambdaInitLM()
        
        # LM optimization
        stop = False
        iter = 0
        last_chi_ = 1e20
        
        while ((stop == False) and (iter < iterations)):
            print(f"iter {iter}, chi = {self.currentChi_}, lambda = {self.currentLambda_}")
            oneStepSuccess = False
            false_cnt = 0
            while ((oneStepSuccess == False) and false_cnt < 10):
                self.SolveLinearSystem()
                
                self.UpdateStates()
                oneStepSuccess = self.IsGoodStepInLM()
                if (oneStepSuccess):
                    self.MakeHessian()
                    false_cnt = 0
                else:
                    false_cnt += 1
                    self.RollbackStates()
                
        
        
    def SetOrdering(self):
        self.ordering_poses_ = 0
        self.ordering_generic_ = 0
        self.ordering_landmarks_ = 0
        
        for _, vertex in self.vertices_.items():
            self.ordering_generic_ += vertex.LocalDimension()
            
            if(self.problem_type == "SLAM_PROBLEM"): self.AddOrderingSLAM(vertex)
        
        # Set landmarks order to bottom
        # Add all_pose_dimension, ensures that poses on top and landmarks on bottom
        if(self.problem_type == "SLAM_PROBLEM"):
            all_pose_dimension = self.ordering_poses_
            for _, landmarkVertex in self.idx_landmark_vertices_.items():
                landmarkVertex.SetOrderingId(landmarkVertex.OrderingId() + all_pose_dimension)
                
            

    def MakeHessian(self):
        size = self.ordering_generic_
        H = torch.zeros([size, size])
        b = torch.zeros(size)
    
    def ComputeLambdaInitLM():
        pass
    
    def SolveLinearSystem(self):
        pass
    
    def UpdateStates(self):
        pass
    
    def IsGoodStepInLM(self):
        pass
    
    def RollbackStates(self):
        pass
    
    def AddOrderingSLAM(self, vertex: Vertex):
        if(self.IsPoseVertex(vertex)):
            vertex.setOrderingId(self.ordering_poses_)
            self.ordering_poses_ += vertex.LocalDimension()
            self.idx_pose_vertices_[vertex.id()] = vertex
            
        elif(self.IsLandmarkVertex(vertex)):
            vertex.setOrderingId(self.ordering_landmarks_)
            self.ordering_landmarks_ += vertex.LocalDimension()
            self.idx_landmark_vertices_[vertex.id()] = vertex
    
    def IsPoseVertex(self, vertex):
        pass
    
    def IsLandmarkVertex(self, vertex):
        pass