package com.zhou.craft.controller;
import com.zhou.craft.dto.RequestForm;import com.zhou.craft.model.CustomRequest;import com.zhou.craft.repo.CustomRequestRepository;import com.zhou.craft.security.*;import org.springframework.http.*;import org.springframework.web.bind.annotation.*;import org.springframework.web.server.ResponseStatusException;import java.util.*;
@RestController @RequestMapping("/api/requests")public class RequestController{
private final CustomRequestRepository repo;public RequestController(CustomRequestRepository repo){this.repo=repo;}
@GetMapping public List<CustomRequest> list(){var p=AuthContext.require();return "USER".equals(p.role())?repo.findByUserIdOrderByCreatedAtDesc(p.id()):repo.findAllByOrderByCreatedAtDesc();}
@PostMapping public CustomRequest create(@RequestBody RequestForm f){var p=AuthContext.require();if(!"USER".equals(p.role()))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"仅普通用户可以发布定制需求");return repo.save(new CustomRequest(p.id(),p.displayName(),f.title(),f.category(),f.budgetMin(),f.budgetMax(),f.description(),f.preferredMaterial(),f.dueDate()));}
}