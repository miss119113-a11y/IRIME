package com.zhou.craft.controller;

import com.zhou.craft.dto.RequestForm;
import com.zhou.craft.model.CustomRequest;
import com.zhou.craft.repo.CustomRequestRepository;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/requests")
public class RequestController {
    private final CustomRequestRepository repo;
    public RequestController(CustomRequestRepository repo){this.repo=repo;}

    @GetMapping
    public List<CustomRequest> list(@RequestParam(required=false) String userName){
        return userName==null||userName.isBlank()?repo.findAllByOrderByCreatedAtDesc():repo.findByUserNameOrderByCreatedAtDesc(userName);
    }

    @PostMapping
    public CustomRequest create(@RequestBody RequestForm f){
        return repo.save(new CustomRequest(f.userName(),f.title(),f.category(),f.budgetMin(),f.budgetMax(),f.description(),f.preferredMaterial(),f.dueDate()));
    }
}
