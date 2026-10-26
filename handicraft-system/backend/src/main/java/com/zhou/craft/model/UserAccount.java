package com.zhou.craft.model;

import jakarta.persistence.*;

@Entity
@Table(name = "sys_user")
public class UserAccount {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(unique = true, nullable = false)
    private String username;
    private String password;
    private String displayName;
    private String role;
    private String avatar;

    public UserAccount() {}
    public UserAccount(String username, String password, String displayName, String role, String avatar) {
        this.username=username; this.password=password; this.displayName=displayName; this.role=role; this.avatar=avatar;
    }
    public Long getId(){return id;}
    public String getUsername(){return username;}
    public String getPassword(){return password;}
    public String getDisplayName(){return displayName;}
    public String getRole(){return role;}
    public String getAvatar(){return avatar;}
}
