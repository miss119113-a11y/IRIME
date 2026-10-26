package com.zhou.craft.controller;

import com.zhou.craft.dto.LoginRequest;
import com.zhou.craft.repo.UserRepository;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private final UserRepository users;
    public AuthController(UserRepository users){this.users=users;}

    @PostMapping("/login")
    public Object login(@RequestBody LoginRequest req){
        return users.findByUsernameAndPassword(req.username(),req.password())
                .<Object>map(u->Map.of("success",true,"id",u.getId(),"username",u.getUsername(),"displayName",u.getDisplayName(),"role",u.getRole(),"avatar",u.getAvatar()))
                .orElseGet(()->Map.of("success",false,"message","用户名或密码错误"));
    }
}
