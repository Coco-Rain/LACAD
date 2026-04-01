import cadquery as cq

torus = cq.Workplane('top').center(2,0).circle(1).revolve(360,(-2,1,0),(-2,0,0))
cyl = cq.Workplane('front').transformed(offset=(0,3,-5)).circle(1).extrude(10)
result = torus.cut(cyl).faces(">Y").fillet(0.3)
cq.exporters.export(result, 'GT.stl')