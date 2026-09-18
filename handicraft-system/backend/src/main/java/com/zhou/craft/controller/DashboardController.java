package com.zhou.craft.controller;

import com.zhou.craft.repo.*;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
@RequestMapping("/api/dashboard")
public class DashboardController {
    private final UserRepository users;private final CraftWorkRepository works;private final CustomRequestRepository requests;private final CraftOrderRepository orders;
    public DashboardController(UserRepository users,CraftWorkRepository works,CustomRequestRepository requests,CraftOrderRepository orders){
        this.users=users;this.works=works;this.requests=requests;this.orders=orders;
    }
    @GetMapping
    public Map<String,Object> dashboard(){
        double revenue=orders.findAll().stream().mapToDouble(o->o.getAmount()).sum();
        return Map.of("users",users.count(),"works",works.count(),"requests",requests.count(),"orders",orders.count(),"revenue",revenue);
    }
}
