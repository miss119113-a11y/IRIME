package com.zhou.craft.model;
import com.fasterxml.jackson.annotation.JsonIgnore;import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="sys_user")
public class UserAccount{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
@Column(unique=true,nullable=false,length=40) private String username;
@JsonIgnore @Column(nullable=false) private String password;
@Column(unique=true,length=120) private String email;
private String displayName; @Column(nullable=false,length=20) private String role; private String avatar; private boolean enabled=true;
private LocalDateTime createdAt;private LocalDateTime updatedAt;
public UserAccount(){} public UserAccount(String username,String password,String email,String displayName,String role,String avatar){this.username=username;this.password=password;this.email=email;this.displayName=displayName;this.role=role;this.avatar=avatar;}
@PrePersist void onCreate(){createdAt=LocalDateTime.now();updatedAt=createdAt;} @PreUpdate void onUpdate(){updatedAt=LocalDateTime.now();}
public Long getId(){return id;} public String getUsername(){return username;} public String getPassword(){return password;} public String getEmail(){return email;} public String getDisplayName(){return displayName;} public String getRole(){return role;} public String getAvatar(){return avatar;} public boolean isEnabled(){return enabled;}
}