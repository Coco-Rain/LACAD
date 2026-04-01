import cadquery as cq

my_plane = cq.Workplane('XZ').plane.rotated((0,10,0))
point_list = [(1,1),(-1,1),(-1,-1),(1,-1)]
wire = cq.Workplane(my_plane).splineApprox(points=point_list, tol=1e-3).close().val()
result = cq.Face.makeFromWires(outerWire=wire)
cq.exporters.export(result, 'GT.stl')